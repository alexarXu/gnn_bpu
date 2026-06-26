import torch
import torch.nn as nn
import numpy as np


# from Rice default 3L setup






# ------------------------
# Time Embedding
# ------------------------
class TimeEmbedding(nn.Module):
    """
    Time embedding using sinusoidal positional encoding
    """

    def __init__(self, device, dim):
        super(TimeEmbedding, self).__init__()
        self.freqs = (2 * np.pi) / (torch.arange(2, dim + 1, 2))
        self.freqs = self.freqs.unsqueeze(0).to(device)

    def forward(self, t):
        self.sin = torch.sin(self.freqs * t)
        self.cos = torch.cos(self.freqs * t)
        return torch.cat([self.sin, self.cos], dim=-1)


class PolynomialEmbedding(nn.Module):
    """
    Polynomial time embedding: [t^1, t^2, t^3, ..., t^tdim]
    With normalization to prevent numerical instability
    """

    def __init__(self, tdim=10, max_time=100.0):
        super(PolynomialEmbedding, self).__init__()
        self.tdim = tdim
        self.max_time = max_time  # Expected maximum time value for normalization

    def forward(self, t):
        # Ensure t has shape [batch_size, 1]
        if t.dim() == 1:
            t = t.unsqueeze(-1)

        # Normalize time to [0, 1] range to prevent large polynomial values
        t_normalized = t / self.max_time
        t_normalized = torch.clamp(t_normalized, 0.0, 1.0)

        # Generate polynomial features: [t^1, t^2, ..., t^tdim]
        poly_features = torch.cat(
            [torch.pow(t_normalized, k) for k in range(1, self.tdim + 1)], dim=-1
        )

        # Optional: further normalize each polynomial feature
        # This ensures all features are in a reasonable range
        poly_features = poly_features / torch.sqrt(
            torch.tensor(self.tdim, dtype=torch.float32, device=t.device)
        )

        return poly_features.to(t.device)


class FourierPositionEmbedding(nn.Module):
    """
    2D Fourier/SinCos position embedding for spatial coordinates.
    Provides rich spatial information for middle fusion with pooled features.
    """

    def __init__(self, H=8, W=8, K=8):
        super().__init__()
        self.H = H
        self.W = W
        self.K = K

        # Create coordinate grids
        us = torch.linspace(-1, 1, W).unsqueeze(0).repeat(H, 1)  # [H, W]
        vs = torch.linspace(-1, 1, H).unsqueeze(1).repeat(1, W)  # [H, W]
        coords = torch.stack([vs, us], dim=-1)  # [H, W, 2]

        # Generate Fourier features with multiple frequencies
        feats = []
        for i in range(K):
            w = 2**i * np.pi
            sin_feat = torch.sin(coords * w)  # [H, W, 2]
            cos_feat = torch.cos(coords * w)  # [H, W, 2]
            feats.append(sin_feat)
            feats.append(cos_feat)

        # Stack all features: [H, W, 2*K*2] = [H, W, 4*K]
        pos_emb = torch.cat(feats, dim=-1)

        # Register as non-trainable parameter
        self.register_buffer("pos_emb", pos_emb)

    def forward(self, batch_size, device):
        """
        Generate position embedding for feature concatenation (middle fusion)

        Args:
            batch_size: Batch size
            device: Device to create tensors on

        Returns:
            Position features [batch_size, feature_dim] for concatenation with pooled features
        """
        # Global average of position embedding across spatial dimensions
        # This gives a summary of spatial information for the entire feature map
        pos_features = self.pos_emb.mean(dim=(0, 1))  # [4*K]

        # Expand for batch
        pos_features = pos_features.unsqueeze(0).expand(batch_size, -1)

        return pos_features.to(device)

    @property
    def feature_dim(self):
        """Number of features for middle fusion"""
        return 4 * self.K


# ------------------------
# DropPath / Stochastic Depth
# ------------------------
class DropPath(nn.Module):
    """
    Drop paths (a.k.a. Stochastic Depth) per sample.
    We linearly increase drop_prob from 0 at block0 to max_drop_prob at blockN.
    """

    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if not self.training or self.drop_prob == 0.0:
            return x
        keep_prob = 1.0 - self.drop_prob
        # shape = [B, 1, 1, 1] to broadcast over C,H,W
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        rand = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        rand.floor_()
        return x.div(keep_prob) * rand


# ------------------------
# SE Block
# ------------------------
class SEBlock(nn.Module):
    """Squeeze-and-Excitation block for channel-wise attention."""

    def __init__(self, channels, reduction=16):
        super().__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # x: [B, C, H, W]
        b, c, _, _ = x.size()
        y = self.fc(x).view(b, c, 1, 1)
        return x * y


# ------------------------
# Residual Block
# ------------------------
class ResidualBlock(nn.Module):
    """Basic 2-layer Residual block with optional SE and DropPath."""

    def __init__(self, channels, use_se=False, reduction_ratio=16, drop_prob=0.0):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.act = nn.ReLU(inplace=True)

        self.use_se = use_se
        if use_se:
            self.se = SEBlock(channels, reduction_ratio)

        self.drop = DropPath(drop_prob)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.use_se:
            out = self.se(out)

        out = self.drop(out)
        out = out + identity
        return self.act(out)


# ------------------------
# CNN Base Class with common methods
# ------------------------
class CNNBase(nn.Module):
    def __init__(
        self,
        use_time_embedding=False,
        time_dim=50,
        time_method="material",
        time_embedding_type="sinusoidal",  # "sinusoidal" or "polynomial"
        use_position_embedding=True,
        position_embedding_type="simple",  # "simple" or "fourier"
        fourier_K=8,  # Number of frequency components for Fourier position embedding
        with_coord=False,  # Add coordinate channels for backward compatibility
    ):
        super().__init__()
        self.use_time_embedding = use_time_embedding
        self.time_dim = time_dim
        self.time_method = time_method  # "material" or "fullmove"
        self.time_embedding_type = time_embedding_type  # "sinusoidal" or "polynomial"

        self.use_position_embedding = use_position_embedding
        self.position_embedding_type = position_embedding_type  # "simple" or "fourier"
        self.fourier_K = fourier_K

        # For backward compatibility with old models that used coordinate channels
        self.with_coord = with_coord

        if self.use_time_embedding:
            # Initialize time embedding (device will be set when model is moved to device)
            self.time_embed = None
            self._time_embed_initialized = False
            # For middle fusion, will create time_to_features in child classes
            self.time_to_features = None
        else:
            self.time_embed = None
            self._time_embed_initialized = False
            self.time_to_features = None

        # Initialize position embedding
        if self.use_position_embedding:
            if self.position_embedding_type == "fourier":
                self.pos_embed = FourierPositionEmbedding(H=8, W=8, K=self.fourier_K)
            else:
                self.pos_embed = (
                    None  # Simple position embedding doesn't need initialization
                )
        else:
            self.pos_embed = None

    def _init_time_embedding(self, device):
        """Initialize time embedding with proper device"""
        if not self._time_embed_initialized and self.use_time_embedding:
            if self.time_embedding_type == "polynomial":
                # For chess, typical game length is 20-100 moves
                max_time = (
                    200.0 if self.time_method == "fullmove" else 100.0
                )  # material count range
                self.time_embed = PolynomialEmbedding(
                    tdim=self.time_dim, max_time=max_time
                )
            else:  # default to sinusoidal
                self.time_embed = TimeEmbedding(device, self.time_dim)
            self._time_embed_initialized = True

    def _init_weights(self):
        """Kaiming init for Conv and Linear, BN weight=1 bias=0"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _get_position_embedding(self, batch_size, device, H=8, W=8):
        """
        Generate position embedding for concatenation with pooled features (middle fusion).

        Args:
            batch_size: Batch size
            device: Device to create tensors on
            H: Height of the feature map (default 8 for chess)
            W: Width of the feature map (default 8 for chess)

        Returns:
            torch.Tensor: Position embedding for concatenation with pooled features
        """
        if not self.use_position_embedding:
            return None

        if self.position_embedding_type == "fourier":
            # Use rich Fourier position embedding
            return self.pos_embed.forward(batch_size, device)
        else:
            # Use simple center coordinates (backward compatibility)
            ys = torch.linspace(-1, 1, H, device=device)
            xs = torch.linspace(-1, 1, W, device=device)
            center_y = ys.mean()
            center_x = xs.mean()
            pos_emb = (
                torch.stack([center_y, center_x], dim=0)
                .unsqueeze(0)
                .expand(batch_size, -1)
            )
            return pos_emb

    def _get_position_features_dim(self):
        """
        Get the dimension of position features for middle fusion.

        Returns:
            int: Dimension of position features
        """
        if not self.use_position_embedding:
            return 0

        if self.position_embedding_type == "fourier":
            return self.pos_embed.feature_dim
        else:
            return 2  # Simple x, y coordinates

    def _process_time_embedding(self, time_values):
        """
        Process time embedding for middle fusion

        Args:
            time_values: Time values to embed [batch_size, 1] or [batch_size]

        Returns:
            Processed time embedding for middle fusion
        """
        if not self.use_time_embedding:
            return None

        # Initialize time embedding if not done
        if not self._time_embed_initialized:
            self._init_time_embedding(time_values.device)

        # Ensure time_values has correct shape
        if time_values.dim() == 1:
            time_values = time_values.unsqueeze(1)

        # Get time embedding
        time_emb = self.time_embed(time_values)  # [batch_size, time_dim]

        # Convert to feature embedding for middle fusion
        if self.time_to_features is None:
            raise RuntimeError(
                "time_to_features layers not initialized. This should be done in child class __init__."
            )

        feature_time = self.time_to_features(time_emb)  # [batch_size, time_dim]
        return feature_time

    def _add_coordinates(self, x):
        """Append two coordinate channels in range [-1,1] for backward compatibility."""
        B, C, H, W = x.shape
        device = x.device
        ys = torch.linspace(-1, 1, H, device=device).view(1, 1, H, 1).expand(B, 1, H, W)
        xs = torch.linspace(-1, 1, W, device=device).view(1, 1, 1, W).expand(B, 1, H, W)
        return torch.cat([x, ys, xs], dim=1)


# ------------------------
# 3-Layer CNN
# ------------------------
class CNN3L(CNNBase):
    def __init__(
        self,
        input_channels=22,
        filter_num=64,
        feat_dim=256,
        reduction_ratio=16,
        drop_out=True,
        dropout_rate=0.2,
        use_time_embedding=False,
        time_dim=50,
        time_method="material",
        time_embedding_type="sinusoidal",
        use_position_embedding=True,
        position_embedding_type="simple",
        fourier_K=8,
        with_coord=False,  # For backward compatibility
    ):
        super().__init__(
            use_time_embedding=use_time_embedding,
            time_dim=time_dim,
            time_method=time_method,
            time_embedding_type=time_embedding_type,
            use_position_embedding=use_position_embedding,
            position_embedding_type=position_embedding_type,
            fourier_K=fourier_K,
            with_coord=with_coord,
        )
        self.filter_num = filter_num
        self.feat_dim = feat_dim
        self.output_dim = feat_dim  # For interface consistency

        # Input channels: original channels + coordinate channels if enabled
        total_ch = input_channels + (2 if self.with_coord else 0)

        self.conv1 = nn.Sequential(
            nn.Conv2d(total_ch, filter_num, kernel_size=3, padding=1),
            nn.BatchNorm2d(filter_num),
            nn.ReLU(),
        )

        self.res_block1 = ResidualBlock(filter_num, use_se=False)
        self.res_block2 = ResidualBlock(
            filter_num, use_se=True, reduction_ratio=reduction_ratio
        )

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.gmp = nn.AdaptiveMaxPool2d(1)

        # Calculate MLP input size for middle fusion
        pooled_features = filter_num * 2  # *2 for avg+max pooling
        pos_dim = self._get_position_features_dim()  # Position embedding dimension

        if self.use_time_embedding:
            # Add time features after pooling
            mlp_input_dim = pooled_features + pos_dim + self.time_dim
        else:
            # No time embedding
            mlp_input_dim = pooled_features + pos_dim

        self.head_mlp = nn.Sequential(
            nn.Linear(mlp_input_dim, feat_dim),
            nn.BatchNorm1d(feat_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate) if drop_out else nn.Identity(),
        )

        # Initialize time embedding layers for middle fusion
        if self.use_time_embedding:
            # For middle fusion, project time to feature space
            self.time_to_features = nn.Sequential(
                nn.Linear(self.time_dim, 256),
                nn.ReLU(inplace=True),
                nn.Linear(256, self.time_dim),
            )

        self._init_weights()
        print(
            f"CNN3L: {total_ch} input channels, {feat_dim} output features, "
            f"pos_embed={self.use_position_embedding}({position_embedding_type}), "
            f"time_embedding={use_time_embedding}, time_method={time_method}, time_type={time_embedding_type}, "
            f"with_coord={with_coord}"
        )

    def forward(self, x, time_values=None):
        # Input: [batch, height, width, channels]
        x = x.permute(0, 3, 1, 2).contiguous()  # → [batch, channels, height, width]

        # Add coordinate channels if enabled (for backward compatibility)
        if self.with_coord:
            x = self._add_coordinates(x)

        # Forward through conv layers
        x = self.conv1(x)
        x = self.res_block1(x)
        x = self.res_block2(x)

        # Global pooling
        avg_pool = self.gap(x).view(x.size(0), -1)
        max_pool = self.gmp(x).view(x.size(0), -1)
        pooled_features = torch.cat([avg_pool, max_pool], dim=1)

        # Add position embedding (middle fusion)
        pos_emb = self._get_position_embedding(
            pooled_features.size(0), pooled_features.device
        )
        if pos_emb is not None:
            pooled_features = torch.cat([pooled_features, pos_emb], dim=1)

        # Handle time embedding (middle fusion)
        if self.use_time_embedding:
            if time_values is None:
                time_values = torch.zeros(
                    pooled_features.size(0), device=pooled_features.device
                )

            time_emb = self._process_time_embedding(time_values)
            # Concatenate time features with pooled features
            pooled_features = torch.cat([pooled_features, time_emb], dim=1)

        return self.head_mlp(pooled_features)



def make_cnn(model_name, **kw):
    if model_name == "CNN3L":
        return CNN3L(**kw)
    elif model_name == "CNN6L":
        return CNN6L(**kw)
    else:
        raise ValueError(f"Model {model_name} not supported")

CNN_MODEL_REGISTRY = {
    'CNN3L': lambda **kw: make_cnn('CNN3L',**kw),
    'CNN6L': lambda **kw: make_cnn('CNN6L',**kw),
}


def get_CNN_model(name: str, **kwargs):
    return CNN_MODEL_REGISTRY[name](
        **kwargs
    )

    
# ------------------------
# 6-Layer CNN
# ------------------------
class CNN6L(CNNBase):
    def __init__(
        self,
        input_channels=22,
        filter_num=64,
        feat_dim=256,
        reduction_ratio=16,
        num_blocks=6,
        max_drop_prob=0.2,
        drop_out=True,
        dropout_rate=0.2,
        use_time_embedding=False,
        time_dim=50,
        time_method="material",
        time_embedding_type="sinusoidal",
        use_position_embedding=True,
        position_embedding_type="simple",
        fourier_K=8,
        with_coord=False,  # For backward compatibility
    ):
        super().__init__(
            use_time_embedding=use_time_embedding,
            time_dim=time_dim,
            time_method=time_method,
            time_embedding_type=time_embedding_type,
            use_position_embedding=use_position_embedding,
            position_embedding_type=position_embedding_type,
            fourier_K=fourier_K,
            with_coord=with_coord,
        )
        self.filter_num = filter_num
        self.feat_dim = feat_dim
        self.output_dim = feat_dim  # For interface consistency

        # Input channels: original channels + coordinate channels if enabled
        total_ch = input_channels + (2 if self.with_coord else 0)

        self.stem = nn.Sequential(
            nn.Conv2d(total_ch, filter_num, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(filter_num),
            nn.ReLU(inplace=True),
            nn.Conv2d(filter_num, filter_num, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(filter_num),
            nn.ReLU(inplace=True),
        )

        self.blocks = nn.ModuleList()
        for i in range(num_blocks):
            drop_p = max_drop_prob * float(i) / float(max(num_blocks - 1, 1))
            use_se = i % 2 == 1
            self.blocks.append(
                ResidualBlock(
                    channels=filter_num,
                    use_se=use_se,
                    reduction_ratio=reduction_ratio,
                    drop_prob=drop_p,
                )
            )

        self.global_pool = nn.AdaptiveAvgPool2d(1)

        # Calculate MLP input size for middle fusion
        pos_dim = self._get_position_features_dim()  # Position embedding dimension

        if self.use_time_embedding:
            # Add time features
            mlp_input_dim = filter_num + pos_dim + self.time_dim
        else:
            # No time embedding
            mlp_input_dim = filter_num + pos_dim

        self.head_mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(mlp_input_dim, feat_dim, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate) if drop_out else nn.Identity(),
        )

        # Initialize time embedding layers for middle fusion
        if self.use_time_embedding:
            # For middle fusion, project time to feature space
            self.time_to_features = nn.Sequential(
                nn.Linear(self.time_dim, 256),
                nn.ReLU(inplace=True),
                nn.Linear(256, self.time_dim),
            )

        self._init_weights()
        print(
            f"CNN6L: {total_ch} input channels, {feat_dim} output features, "
            f"pos_embed={self.use_position_embedding}({position_embedding_type}), "
            f"time_embedding={use_time_embedding}, time_method={time_method}, time_type={time_embedding_type}, "
            f"with_coord={with_coord}"
        )

    def forward(self, x, time_values=None):
        # x: [B, H, W, C]
        x = x.permute(0, 3, 1, 2).contiguous()  # → [B, C, H, W]

        # Add coordinate channels if enabled (for backward compatibility)
        if self.with_coord:
            x = self._add_coordinates(x)

        # Forward through network
        x = self.stem(x)
        for blk in self.blocks:
            x = blk(x)

        # Global pooling
        x = self.global_pool(x)
        pooled_features = x.flatten(1)

        # Add position embedding (middle fusion)
        pos_emb = self._get_position_embedding(
            pooled_features.size(0), pooled_features.device
        )
        if pos_emb is not None:
            pooled_features = torch.cat([pooled_features, pos_emb], dim=1)

        # Handle time embedding (middle fusion)
        if self.use_time_embedding:
            if time_values is None:
                time_values = torch.zeros(
                    pooled_features.size(0), device=pooled_features.device
                )

            time_emb = self._process_time_embedding(time_values)
            # Concatenate time features with pooled features
            pooled_features = torch.cat([pooled_features, time_emb], dim=1)

        return self.head_mlp(pooled_features)
