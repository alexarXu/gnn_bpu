import torch
import torch.nn as nn

RNN_MODELS = ['BaseRNN']

class BaseRNN(nn.Module):
    def __init__(self,
                 W_init,
                 input_dim: int, # the size of input that came into RNN
                 sensory_idx,
                 KC_idx,
                 internal_idx,
                 output_idx,
                 learnable: bool = False,
                 dropout_rate: float = 0.2,
                 use_residual: bool = False,
                 timesteps: int = 5,
                 learnable_type = None,
                 no_input_proj = False # if the input size is designed to be 430 (our sensory dim size)
                 ):
        super().__init__()
        self.register_buffer("sensory_idx", torch.tensor(sensory_idx, dtype=torch.long))
        self.KC_idx = KC_idx
        self.internal_idx = internal_idx
        self.output_idx = output_idx
        self.output_dim = len(self.output_idx)
        self.total_dim = len(self.sensory_idx) + len(self.KC_idx) + len(self.internal_idx) + len(self.output_idx)
        self.use_residual = use_residual

        W_init_tensor = torch.tensor(W_init, dtype=torch.float32)
        if learnable:
            mask = torch.zeros_like(W_init_tensor)
            if learnable_type == 'betweenKC':# train any connection that has one end in KC, changing weights = 3244
                non_zero_mask = torch.tensor(W_init != 0, dtype=torch.float32)
                mask[self.KC_idx, :] = non_zero_mask[self.KC_idx, :]
                mask[:, self.KC_idx] = non_zero_mask[:, self.KC_idx]
            elif learnable_type == 'withinKC': # train the whole 144 * 144, changing weights = 20736
                KC_idx_tensor = torch.tensor(self.KC_idx)
                rows, cols = torch.meshgrid(KC_idx_tensor, KC_idx_tensor, indexing='ij')
                mask[rows, cols] = 1.0
            elif learnable_type == 'all': # train all non-zero in the W
                mask = torch.tensor(W_init != 0, dtype=torch.float32)
            else:
                raise ValueError("Not supported learnable_type")
            self.register_buffer('mask', mask)

            self.W = nn.Parameter(W_init_tensor)
            self.W.register_hook(lambda grad: grad * self.mask)
        else:# fix the weight
            self.register_buffer('W', W_init_tensor)

        if no_input_proj:
            assert input_dim == len(sensory_idx)
            self.input_proj = nn.Identity()
        else:
            self.input_proj = nn.Linear(input_dim, len(self.sensory_idx))

        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.timesteps = timesteps

    def forward(self, x):
        batch_size, device = x.shape[0], x.device

        if x.ndim == 3:
            multi_dim = True
            T = x.shape[1] + 1
        else:
            multi_dim = False
            T = self.timesteps
            assert T is not None

        h_state = torch.zeros(batch_size, self.total_dim, device=device)
        for t in range(T):
            if multi_dim:
                try:
                    init = torch.zeros((batch_size, self.total_dim), dtype=x.dtype, device=x.device)
                    E = self.dropout(self.input_proj(x[:,t,:]))
                    E_t = init.scatter_add_(dim=1, index=self.sensory_idx.expand(batch_size, -1).to(E.device), src=E)
                except:
                    E_t = torch.zeros((batch_size, self.total_dim), dtype=x.dtype, device=x.device)
            elif t == 0:
                init = torch.zeros((batch_size, self.total_dim), dtype=x.dtype, device=x.device)
                x = x.reshape(x.size(0), -1)
                E = self.dropout(self.input_proj(x))
                E_t = init.scatter_add_(dim=1, index=self.sensory_idx.expand(batch_size, -1).to(E.device), src=E)
            else:
                E_t = torch.zeros((batch_size, self.total_dim), dtype=x.dtype, device=x.device)

            h_next = h_state @ self.W + E_t
            if self.use_residual:
                h_next = h_state + h_next
            h_next = self.activation(h_next)
            h_state = h_next

        out = h_state[:, self.output_idx]
        return out


