import os

import torch
from torch import Tensor
from torch.nn import Linear
from typing import Union
from torch_geometric.nn import MessagePassing
from torch_geometric.typing import Adj, OptPairTensor, OptTensor, Size
from torch_geometric.nn.inits import reset

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

_LAYER_SEED_STEP = 10_007
_LAYER_INDEX = {
    'sensory_hidden': 0,
    'hidden_output': 1,
    'input_output': 0,
}


def _layer_init_seed(base_seed, sparse_prob_type):
    if base_seed is None:
        return None
    layer_idx = _LAYER_INDEX.get(sparse_prob_type, 0)
    return int(base_seed) + layer_idx * _LAYER_SEED_STEP


def _make_rng(init_seed):
    if init_seed is None:
        return np.random.default_rng()
    return np.random.default_rng(int(init_seed))


def _hidden_idx(conn):
    return conn['KC_idx'] + conn['internal_idx']


def _connectome_block(conn, sparse_prob_type):
    init = conn['W']
    if sparse_prob_type == 'input_output':
        return init[np.ix_(conn['sensory_idx'], conn['output_idx'])]
    if sparse_prob_type == 'sensory_hidden':
        hidden = _hidden_idx(conn)
        return init[np.ix_(conn['sensory_idx'], hidden)]
    if sparse_prob_type == 'hidden_output':
        hidden = _hidden_idx(conn)
        return init[np.ix_(hidden, conn['output_idx'])]
    if sparse_prob_type == 'all_all':
        return init
    raise NotImplementedError(f'invalid sparsity type: {sparse_prob_type}')


def _random_sparsity(sparse_prob_type, conn):
    block = _connectome_block(conn, sparse_prob_type)
    nnz = np.count_nonzero(block)
    return 1.0 - nnz / block.size


def verify_connectome_layers(conn, init_type='droso'):
    """Compare connectome blocks with SparseLinear / Connectome2Layer properties."""
    W = conn['W']
    s, o = conn['sensory_idx'], conn['output_idx']
    hidden = _hidden_idx(conn)

    blocks = {
        'input_output': W[np.ix_(s, o)],
        'sensory_hidden': W[np.ix_(s, hidden)],
        'hidden_output': W[np.ix_(hidden, o)],
    }
    report = {'full_W': {
        'shape': W.shape,
        'nnz': int(np.count_nonzero(W)),
        'sparsity': float(1 - np.count_nonzero(W) / W.size),
        'rank': int(np.linalg.matrix_rank(W)),
    }}
    for name, block in blocks.items():
        report[name] = {
            'shape': block.shape,
            'linear_weight_shape': (block.shape[1], block.shape[0]),  # nn.Linear [out, in]
            'nnz': int(np.count_nonzero(block)),
            'sparsity': float(1 - np.count_nonzero(block) / block.size),
            'rank': int(np.linalg.matrix_rank(block)),
            'random_init_sparsity': _random_sparsity(name, conn),
        }

    # effective rank of 2-layer path (sensory -> hidden -> output)
    path_rank = int(np.linalg.matrix_rank(blocks['sensory_hidden'] @ blocks['hidden_output']))
    report['2layer_path'] = {
        'layer1_shape': blocks['sensory_hidden'].shape,
        'layer2_shape': blocks['hidden_output'].shape,
        'effective_rank': path_rank,
        'layer1_nnz': report['sensory_hidden']['nnz'],
        'layer2_nnz': report['hidden_output']['nnz'],
    }
    report['init_type'] = init_type
    return report


def _randomize_mask_preserve_degrees(mask, rng, n_swaps=50_000):
    """Edge-swap randomization: preserve row/col sums (fan-out / fan-in)."""
    mask = mask.astype(bool).copy()
    rows, cols = np.where(mask)
    n_edges = len(rows)
    if n_edges < 2:
        return mask

    edges = np.column_stack([rows, cols])
    for _ in range(n_swaps):
        i, j = rng.integers(0, n_edges, size=2)
        if i == j:
            continue
        r1, c1 = edges[i]
        r2, c2 = edges[j]
        if r1 == r2 or c1 == c2:
            continue
        if mask[r1, c2] or mask[r2, c1]:
            continue
        mask[r1, c1] = False
        mask[r2, c2] = False
        mask[r1, c2] = True
        mask[r2, c1] = True
        edges[i] = (r1, c2)
        edges[j] = (r2, c1)
    return mask


def _matrix_rank_tol(M, rtol=1e-4):
    rows = np.any(M != 0, axis=1)
    cols = np.any(M != 0, axis=0)
    if not rows.any() or not cols.any():
        return 0
    sub = M[np.ix_(rows, cols)]
    s = np.linalg.svd(sub, compute_uv=False)
    if s.size == 0:
        return 0
    return int((s > rtol * s[0]).sum())


def _scale_nonzero_weights(
        weight,
        block,
        match_connectome_weight_range=False,
        custom_weight_scale=None,
        default_weight_scale=100.0,
):
    """Rescale non-zero entries: match connectome stats, custom std, or default std."""
    weight = np.array(weight, dtype=np.float32, copy=True)
    nz = weight != 0
    if not nz.any():
        return weight
    vals = weight[nz].astype(np.float64)
    if match_connectome_weight_range:
        ref = block[block != 0].astype(np.float64)
        ref_std = ref.std() + 1e-8
        vals = (vals - vals.mean()) / (vals.std() + 1e-8) * ref_std + ref.mean()
    elif custom_weight_scale is not None:
        vals = (vals - vals.mean()) / (vals.std() + 1e-8) * float(custom_weight_scale)
    else:
        vals = (vals - vals.mean()) / (vals.std() + 1e-8) * float(default_weight_scale)
    weight[nz] = vals.astype(np.float32)
    return weight


def _weight_nz_stats(weight):
    nz = weight[weight != 0]
    if nz.size == 0:
        return {'min': 0.0, 'max': 0.0, 'mean': 0.0, 'std': 0.0}
    return {
        'min': float(nz.min()),
        'max': float(nz.max()),
        'mean': float(nz.mean()),
        'std': float(nz.std()),
    }


def _init_random_sparsity_rank(block, rng=None):
    """Random weights: matched nnz; spectrum from connectome SVD."""
    rng = rng or np.random.default_rng()
    target_nnz = int(np.count_nonzero(block))
    target_rank = int(np.linalg.matrix_rank(block.astype(np.float64)))
    out_f, in_f = block.shape[1], block.shape[0]
    r = max(1, min(target_rank, out_f, in_f))

    _, s0, _ = np.linalg.svd(block.astype(np.float64), full_matrices=False)
    s_use = s0[:r]
    Q_out, _ = np.linalg.qr(rng.standard_normal((out_f, r)))
    Q_in, _ = np.linalg.qr(rng.standard_normal((in_f, r)))
    dense = (Q_out @ np.diag(s_use) @ Q_in.T).astype(np.float32)
    flat = dense.ravel()
    keep = np.argpartition(np.abs(flat), -target_nnz)[-target_nnz:]
    weight = np.zeros(flat.size, dtype=np.float32)
    weight[keep] = flat[keep]
    return weight.reshape(out_f, in_f)


def _init_random_sparsity_degree(block, rng=None):
    """Random mask with matched row/col degree sequences; N(0,1) values."""
    rng = rng or np.random.default_rng()
    ref_mask = (block != 0).T
    mask = _randomize_mask_preserve_degrees(ref_mask, rng)
    weight = np.zeros(ref_mask.shape, dtype=np.float32)
    weight[mask] = rng.standard_normal(int(mask.sum())).astype(np.float32)
    return weight


def _init_weight_from_block(
        block,
        init_type,
        match_connectome_weight_range=False,
        custom_weight_scale=None,
        default_weight_scale=100.0,
        init_seed=None,
):
    """Build nn.Linear weight [out, in] from a connectome block [rows, cols]."""
    rng = _make_rng(init_seed)
    random_init_types = {
        'random', 'random_sparsity', 'random_sparsity_rank', 'random_sparsity_degree',
    }

    if init_type == 'droso':
        weight = torch.from_numpy(block.astype(np.float32)).float().t().contiguous()
    elif init_type == 'droso_permute':
        temp = block.flatten()
        temp = rng.permutation(temp)
        temp = temp.reshape(block.shape)
        weight = torch.from_numpy(temp.astype(np.float32)).float().t().contiguous()
    elif init_type == 'random':
        rand_block = np.zeros_like(block, dtype=np.float32)
        mask = block != 0
        rand_block[mask] = rng.standard_normal(int(mask.sum())).astype(np.float32)
        w = _scale_nonzero_weights(
            rand_block.T, block,
            match_connectome_weight_range, custom_weight_scale, default_weight_scale,
        )
        weight = torch.from_numpy(w)
    elif init_type == 'random_sparsity':
        target_nnz = int(np.count_nonzero(block))
        out_f, in_f = block.shape[1], block.shape[0]
        flat = np.zeros(out_f * in_f, dtype=np.float32)
        idx = rng.choice(flat.size, target_nnz, replace=False)
        flat[idx] = rng.standard_normal(target_nnz).astype(np.float32)
        w = _scale_nonzero_weights(
            flat.reshape(out_f, in_f), block,
            match_connectome_weight_range, custom_weight_scale, default_weight_scale,
        )
        weight = torch.from_numpy(w)
    elif init_type == 'random_sparsity_rank':
        w = _init_random_sparsity_rank(block, rng)
        w = _scale_nonzero_weights(
            w, block, match_connectome_weight_range, custom_weight_scale, default_weight_scale,
        )
        weight = torch.from_numpy(w)
    elif init_type == 'random_sparsity_degree':
        w = _init_random_sparsity_degree(block, rng)
        w = _scale_nonzero_weights(
            w, block, match_connectome_weight_range, custom_weight_scale, default_weight_scale,
        )
        weight = torch.from_numpy(w)
    elif init_type in random_init_types:
        raise ValueError(f'unhandled init_type: {init_type}')
    else:
        raise ValueError(f'unknown init_type: {init_type}')
    return weight


def compare_connectome_networks(
        conn,
        connectome_type='2layer',
        init_ref='droso',
        init_like='random',
        match_connectome_weight_range=False,
        custom_weight_scale=None,
        default_weight_scale=100.0,
        init_seed=None,
):
    """Compare reference connectome vs connectome-like network.

    init_like='random': same mask as connectome, random values.
    init_like='random_sparsity': random mask, matched nnz/sparsity.
    init_like='random_sparsity_rank': matched nnz + rank (<= connectome).
    init_like='random_sparsity_degree': matched nnz + row/col degree sequences.
    """
    wr_kw = dict(
        match_connectome_weight_range=match_connectome_weight_range,
        custom_weight_scale=custom_weight_scale,
        default_weight_scale=default_weight_scale,
        init_seed=init_seed,
    )
    if connectome_type == '2layer':
        ref = Connectome2Layer(conn, init_type=init_ref, trainable=False)
        like = Connectome2Layer(conn, init_type=init_like, trainable=False, **wr_kw)
        layers = [
            ('L1 sensory->hidden', ref.layer1, like.layer1, 'sensory_hidden'),
            ('L2 hidden->output', ref.layer2, like.layer2, 'hidden_output'),
        ]
    else:
        ref = SparseLinear(
            len(conn['sensory_idx']), len(conn['output_idx']),
            sparse_prob_type='input_output', init_type=init_ref, conn=conn,
        )
        like = SparseLinear(
            len(conn['sensory_idx']), len(conn['output_idx']),
            sparse_prob_type='input_output', init_type=init_like, conn=conn, **wr_kw,
        )
        layers = [('sensory->output', ref, like, 'input_output')]

    rows = []
    for label, mod_ref, mod_like, block_name in layers:
        w_ref = mod_ref.linear.weight.detach()
        w_like = mod_like.linear.weight.detach()
        mask_ref = (w_ref != 0).numpy()
        mask_like = (w_like != 0).numpy()
        block = _connectome_block(conn, block_name)
        w_ref_np = w_ref.numpy()
        w_like_np = w_like.numpy()
        rank_ref = _matrix_rank_tol(w_ref_np)
        rank_like = _matrix_rank_tol(w_like_np)
        row_ref, row_like = mask_ref.sum(axis=1), mask_like.sum(axis=1)
        col_ref, col_like = mask_ref.sum(axis=0), mask_like.sum(axis=0)
        rows.append({
            'layer': label,
            'shape_ref': tuple(w_ref.shape),
            'shape_like': tuple(w_like.shape),
            'shape_match': w_ref.shape == w_like.shape,
            'nnz_ref': int(mask_ref.sum()),
            'nnz_like': int(mask_like.sum()),
            'nnz_connectome': int(np.count_nonzero(block)),
            'sparsity_ref': float(1 - mask_ref.mean()),
            'sparsity_like': float(1 - mask_like.mean()),
            'mask_iou': float((mask_ref & mask_like).sum() / (mask_ref | mask_like).sum()),
            'rank_ref': rank_ref,
            'rank_like': rank_like,
            'rank_connectome': _matrix_rank_tol(block.astype(np.float64)),
            'rank_ok': rank_like <= _matrix_rank_tol(block.astype(np.float64)),
            'row_degree_match': bool(np.array_equal(row_ref, row_like)),
            'col_degree_match': bool(np.array_equal(col_ref, col_like)),
            'weight_std_ref': _weight_nz_stats(w_ref_np)['std'],
            'weight_std_like': _weight_nz_stats(w_like_np)['std'],
            'weight_min_like': _weight_nz_stats(w_like_np)['min'],
            'weight_max_like': _weight_nz_stats(w_like_np)['max'],
            'weight_corr': float(np.corrcoef(w_ref_np.ravel(), w_like_np.ravel())[0, 1])
            if w_ref.numel() > 1 else 0.0,
        })
    return rows


def audit_trainable_params(model):
    """Return per-parameter train/freeze status."""
    rows = []
    for name, p in model.named_parameters():
        rows.append({
            'parameter': name,
            'requires_grad': bool(p.requires_grad),
            'frozen': not p.requires_grad,
            'numel': p.numel(),
        })
    return rows


def connectome_head_frozen(model):
    """True if all Connectome2Layer / SparseLinear weights are frozen."""
    for name, p in model.named_parameters():
        if 'sparse_fc' in name and 'weight' in name:
            if p.requires_grad:
                return False
    return True


def save_connectome_weights_csv(module, path, include_zeros=False):
    """Export Connectome2Layer, SparseLinear, or model.sparse_fc weights to CSV."""
    if hasattr(module, 'sparse_fc'):
        module = module.sparse_fc

    if isinstance(module, Connectome2Layer):
        layer_modules = [
            ('L1_sensory_hidden', module.layer1),
            ('L2_hidden_output', module.layer2),
        ]
    elif isinstance(module, SparseLinear):
        layer_modules = [('sparse_linear', module)]
    else:
        raise TypeError(
            'module must be Connectome2Layer, SparseLinear, or have a sparse_fc attribute'
        )

    rows = []
    for layer_name, sparse_layer in layer_modules:
        w = sparse_layer.linear.weight.detach().cpu().numpy()
        if include_zeros:
            out_idx, in_idx = np.indices(w.shape)
            for o, i, val in zip(out_idx.ravel(), in_idx.ravel(), w.ravel()):
                rows.append({
                    'layer': layer_name,
                    'out_idx': int(o),
                    'in_idx': int(i),
                    'weight': float(val),
                })
        else:
            nz = np.nonzero(w)
            for o, i in zip(nz[0], nz[1]):
                rows.append({
                    'layer': layer_name,
                    'out_idx': int(o),
                    'in_idx': int(i),
                    'weight': float(w[o, i]),
                })

    import pandas as pd
    out_path = os.path.abspath(path)
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    pd.DataFrame(rows).to_csv(out_path, index=False)
    return out_path


class SparseLinear(nn.Module):
    def __init__(
            self,
            in_features: int,
            out_features: int,
            sparse_prob_type = None,
            init_type = None,
            trainable: bool = False,
            conn = None,
            match_connectome_weight_range=False,
            custom_weight_scale=None,
            default_weight_scale=100.0,
            init_seed=None,
    ):
        super().__init__()
        layer_seed = _layer_init_seed(init_seed, sparse_prob_type)
        wr_kw = dict(
            match_connectome_weight_range=match_connectome_weight_range,
            custom_weight_scale=custom_weight_scale,
            default_weight_scale=default_weight_scale,
            init_seed=layer_seed,
        )

        if init_type == 'random':
            block = _connectome_block(conn, sparse_prob_type)
            weight = _init_weight_from_block(block, 'random', **wr_kw)
            self.linear = nn.Linear(weight.size(1), weight.size(0), bias=False)
            with torch.no_grad():
                self.linear.weight.copy_(weight)
            nonzeros = torch.count_nonzero(self.linear.weight).item()
            print(f"Random init [SparseLinear] non-zero weights: {nonzeros} (mask from connectome)")

        elif init_type in ('random_sparsity', 'random_sparsity_rank', 'random_sparsity_degree'):
            block = _connectome_block(conn, sparse_prob_type)
            weight = _init_weight_from_block(block, init_type, **wr_kw)
            self.linear = nn.Linear(weight.size(1), weight.size(0), bias=False)
            with torch.no_grad():
                self.linear.weight.copy_(weight)
            target = int(np.count_nonzero(block))
            nonzeros = torch.count_nonzero(self.linear.weight).item()
            print(f"[{init_type}] SparseLinear nnz={nonzeros} (target {target})")

        else:
            block = _connectome_block(conn, sparse_prob_type)
            weight = _init_weight_from_block(block, init_type, init_seed=layer_seed)

            self.linear = nn.Linear(
                in_features=weight.size(1),
                out_features=weight.size(0),
                bias=False
            )
            with torch.no_grad():
                self.linear.weight.copy_(weight)

        if not trainable:
            self.linear.weight.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class Connectome2Layer(nn.Module):
    """Two SparseLinear layers: sensory -> hidden (KC+internal) -> output."""

    def __init__(
            self,
            conn,
            init_type='droso',
            trainable=False,
            match_connectome_weight_range=False,
            custom_weight_scale=None,
            default_weight_scale=100.0,
            init_seed=None,
    ):
        super().__init__()
        hidden = _hidden_idx(conn)
        conn = {**conn, 'hidden_idx': hidden}
        wr_kw = dict(
            match_connectome_weight_range=match_connectome_weight_range,
            custom_weight_scale=custom_weight_scale,
            default_weight_scale=default_weight_scale,
            init_seed=init_seed,
        )
        self.layer1 = SparseLinear(
            len(conn['sensory_idx']), len(hidden),
            sparse_prob_type='sensory_hidden',
            init_type=init_type,
            trainable=trainable,
            conn=conn,
            **wr_kw,
        )
        self.layer2 = SparseLinear(
            len(hidden), len(conn['output_idx']),
            sparse_prob_type='hidden_output',
            init_type=init_type,
            trainable=trainable,
            conn=conn,
            **wr_kw,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.layer1(x))
        return F.relu(self.layer2(x))


class CustomV1(MessagePassing):
    def __init__(self, nn: torch.nn.Module, eps: float = 0.,
                 train_eps: bool = False, edge_dim: int = None,
                 **kwargs):
        kwargs.setdefault('aggr', 'add')
        super().__init__(**kwargs)
        self.nn = nn
        
        self.initial_eps = eps
        if train_eps:
            self.eps = torch.nn.Parameter(torch.empty(1))
        else:
            self.register_buffer('eps', torch.empty(1))

        assert edge_dim is not None
        if isinstance(self.nn, torch.nn.Sequential):
            nn = self.nn[0]
        if hasattr(nn, 'in_features'):
            hidden_dim = nn.in_features
        elif hasattr(nn, 'in_channels'):
            hidden_dim = nn.in_channels
        else:
            raise ValueError("Could not infer input channels from `nn`.")
        self.lin = Linear(edge_dim, hidden_dim)
        self.message_mlp = Linear(2 * hidden_dim, hidden_dim)
        
        self.reset_parameters()

    def reset_parameters(self):
        reset(self.nn)
        self.eps.data.fill_(self.initial_eps)
        if self.lin is not None:
            self.lin.reset_parameters()
        self.message_mlp.reset_parameters()

    def forward(
        self,
        x: Union[Tensor, OptPairTensor],
        edge_index: Adj,
        edge_attr: OptTensor = None,
        size: Size = None,
    ) -> Tensor:
        if isinstance(x, Tensor):
            x = (x, x)

        # propagate_type: (x: OptPairTensor, edge_attr: OptTensor)
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr, size=size)

        x_r = x[1]
        if x_r is not None:
            out = out + (1 + self.eps) * x_r

        return self.nn(out)

    def message(self, x_j: Tensor, edge_attr: Tensor) -> Tensor:
        if self.lin is None and x_j.size(-1) != edge_attr.size(-1):
            raise ValueError("Node and edge feature dimensionalities do not match.")

        if self.lin is not None:
            edge_attr = self.lin(edge_attr)

        # Concatenate node and edge features
        combined = torch.cat([x_j, edge_attr], dim=-1)
        # Pass through message MLP and apply ReLU
        return torch.relu(self.message_mlp(combined))

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(nn={self.nn})'
    





class CustomV2(MessagePassing):
    def __init__(self, nn: torch.nn.Module, eps: float = 0.,
                 train_eps: bool = False, edge_dim: int = None,
                 project_edge = True,
                 **kwargs):
        kwargs.setdefault('aggr', 'add')
        super().__init__(**kwargs)
        self.nn = nn
        
        self.initial_eps = eps
        if train_eps:
            self.eps = torch.nn.Parameter(torch.empty(1))
        else:
            self.register_buffer('eps', torch.empty(1))

        assert edge_dim is not None
        if isinstance(self.nn, torch.nn.Sequential):
            nn = self.nn[0]
        if hasattr(nn, 'in_features'):
            hidden_dim = nn.in_features
        elif hasattr(nn, 'in_channels'):
            hidden_dim = nn.in_channels
        else:
            raise ValueError("Could not infer input channels from `nn`.")

        if (edge_dim != hidden_dim) or project_edge:
            self.lin = Linear(edge_dim, hidden_dim)
        else:
            self.lin = None

        self.message_mlp = torch.nn.Sequential(
            Linear(3 * hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            Linear(hidden_dim, hidden_dim)
        )
        
        self.reset_parameters()

    def reset_parameters(self):
        reset(self.nn)
        self.eps.data.fill_(self.initial_eps)
        if self.lin is not None:
            self.lin.reset_parameters()
        for layer in self.message_mlp:
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()

    def forward(
        self,
        x: Union[Tensor, OptPairTensor],
        edge_index: Adj,
        edge_attr: OptTensor = None,
        size: Size = None,
    ) -> Tensor:
        if isinstance(x, Tensor):
            x = (x, x)

        out = self.propagate(edge_index, x=x, edge_attr=edge_attr, size=size)

        x_r = x[1]
        if x_r is not None:
            out = out + (1 + self.eps) * x_r

        return self.nn(out)

    def message(self, x_i: Tensor, x_j: Tensor, edge_attr: Tensor) -> Tensor:
        edge_feat = self.lin(edge_attr) if self.lin is not None else edge_attr
        combined = torch.cat([x_i, x_j, edge_feat], dim=-1)
        return (x_j + edge_feat + self.message_mlp(combined)).relu()

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(nn={self.nn})'
    


