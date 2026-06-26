import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric
from torch_geometric.nn import (
    GATv2Conv, TransformerConv,
    GINEConv, GENConv,ResGatedGraphConv,
    global_mean_pool, global_max_pool, GCNConv, ChebConv, GraphConv,
)
from src.chess_utils.tokenizer import EDGE_DIM as _EDGE_DIM, F_DIM as _F_DIM, HUB as _GLOBAL_ID,N_NODES as _N_NODES
from src.model.custom_layers import CustomV1,CustomV2

GNN_ATTENTION_MODELS = ['GATv2','Transf']
GNN_WEIGHT_MODELS = ['GCN','Cheb','Graph',]
GNN_ATTRI_MODELS = ['GINE','GEN','ResGateGraph','CustomV1','CustomV2']


def _resolve_activation(act: str):
    act = act.lower()
    if act == 'gelu':
        return nn.GELU(), F.gelu
    if act == 'elu':
        return nn.ELU(), F.elu
    # default
    return nn.ReLU(), F.relu

# for models taking edge attribute
class Edge_Up(nn.Module):
    def __init__(self, edge_in: int, hidden_dim: int, act_layer: nn.Module):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(edge_in, hidden_dim), act_layer,
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, ea):
        return self.mlp(ea)

# for models taking edge weight
class Edge_Down(nn.Module):
    def __init__(self, edge_in: int, act_layer: nn.Module):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(edge_in, 1), act_layer,
        )

    def forward(self, ea):
        return self.mlp(ea)

class EdgeUpdater(nn.Module):
    def __init__(self, hidden_dim: int, edge_in: int, act_layer: nn.Module):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim + edge_in, hidden_dim),
            act_layer,
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, x, edge_index, edge_attr):
        row, col = edge_index
        edge_in = torch.cat([x[col], x[row], edge_attr], dim=-1)
        return self.mlp(edge_in)


class FiLM(nn.Module):
    def __init__(self, ctx_dim, hidden_dim, act_layer: nn.Module):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(ctx_dim, hidden_dim * 2),
            act_layer,
            nn.Linear(hidden_dim * 2, hidden_dim * 2)
        )

    def forward(self, h, batch, ctx):
        temp = self.mlp(ctx)  # [batch, 2H]
        gamma, beta = temp.chunk(2, dim=-1)
        gamma, beta = gamma[batch], beta[batch]
        return gamma * h + beta

class BaseEncoder(nn.Module):
    def __init__(
        self,
        conv_class,
        in_feats:      int,
        edge_in_feats: int,
        hidden_dim:    int = 64,
        num_layer:     int = 2,
        dropout:       float = 0.1,
        edge_dropout:  float = 0.1,
        heads:         int  = None,     # only used by attention layers
        residual:      bool = True,
        gated:         bool = False,
        update_edge:   bool = False,
        return_all_hiddens: bool = False,
        graph_norm:    bool = False,
        batch_norm:    bool = False,
        use_film:      bool = False,
        return_before_pool: bool = False,
        act: str = 'relu',
        NodePool: bool = False,
        NodeFdim: int = None,
        MLPFeaturePool: bool = False,
        return_for_cnn: bool = False,
    ):
        super().__init__()
        self.NodeFdim = NodeFdim
        if self.NodeFdim is not None:
            assert not return_before_pool
            assert not return_all_hiddens
            assert not NodePool is None
            hidden_dim = hidden_dim + NodeFdim

        self.conv_class = conv_class
        self.hidden_dim = hidden_dim
        self.heads      = heads
        self.residual = residual
        self.gated = gated
        self.return_all_hiddens = return_all_hiddens
        self.edge_dropout = edge_dropout
        self.return_before_pool = return_before_pool
        self.NodePool = NodePool
        self.MLPFeaturePool = MLPFeaturePool
        self.return_for_cnn = return_for_cnn

        if self.NodePool:
            assert not return_before_pool
            assert not return_all_hiddens
            self.node_mlp = nn.Linear(hidden_dim, self.NodePool)

        if self.MLPFeaturePool :
            assert not return_before_pool
            assert not return_all_hiddens
            self.feature_mlp_weight = nn.Parameter(torch.randn(hidden_dim, _N_NODES))  # shape [256,65]
            self.feature_mlp_bias = nn.Parameter(torch.zeros(hidden_dim))  # shape [256]


        self.act_layer, self.act_fn = _resolve_activation(act)

        if gated:
            assert residual, "Gated residual requires residual=True"
            self.alpha = nn.Parameter(torch.zeros(num_layer))  #  Re-Zero

        # pick edge embedder --------------------------------------------------
        if update_edge:
            self.edge_mode = 'update'
            self.edge_updaters = nn.ModuleList(
                [EdgeUpdater(hidden_dim, (_EDGE_DIM if i == 0 else hidden_dim), self.act_layer) for i in
                 range(num_layer)]
            )
            self.edge_bn = nn.ModuleList(nn.LayerNorm(hidden_dim) for _ in range(num_layer))

        elif conv_class in {GCNConv, ChebConv, GraphConv}:
            self.edge_emb = Edge_Down(edge_in_feats, self.act_layer)
            self.edge_mode = 'weight'
        else:
            self.edge_emb = Edge_Up(edge_in_feats, hidden_dim, self.act_layer)
            self.edge_mode = 'attr'

        # node stem -----------------------------------------------------------
        self.input_proj = nn.Linear(in_feats, hidden_dim)
        self.drop       = nn.Dropout(dropout)


        # stack of conv + BN --------------------------------------------------
        self.convs, self.bns = nn.ModuleList(), nn.ModuleList()
        for _ in range(num_layer):
            self.convs.append(self._make_layer(conv_class, hidden_dim))
            if graph_norm:
                self.bns.append(torch_geometric.nn.norm.GraphNorm(hidden_dim))
            elif batch_norm:
                self.bns.append(nn.BatchNorm1d(hidden_dim))
            else:
                self.bns.append(nn.LayerNorm(hidden_dim))

        if use_film:
            self.films = nn.ModuleList(FiLM(hidden_dim, hidden_dim, self.act_layer) for _ in range(num_layer))
        else:
            self.films = None

    # --------------------------------------------------------------------- #
    def _make_layer(self, cls, h):
        def _node_mlp():
            return nn.Sequential(nn.Linear(h, h), self.act_layer, nn.Linear(h, h))

        if cls in {GINEConv, CustomV1}:
            return cls(_node_mlp(), edge_dim=h)
        if cls in {CustomV2}:
            project_edge=(self.edge_mode != 'update')
            print("Using CustomV2 with project_edge = ", project_edge)
            return cls(_node_mlp(), edge_dim=h, project_edge=project_edge)
        if cls in {GATv2Conv, TransformerConv}:
            return cls(h, h, heads=self.heads, edge_dim=h, concat=False)
        if cls is GENConv:
            return cls(h, h, edge_dim=h)
        if cls is ChebConv:
            return cls(h, h, K=3)
        if cls is ResGatedGraphConv:
            return cls(h, h, edge_dim=h)
        return cls(h, h)


    def forward(self, data):
        x, ei, ea, batch = data.x, data.edge_index, data.edge_attr, data.batch
        x = self.input_proj(x)
        
        # edge dropout during training
        if self.training and hasattr(self, 'edge_dropout') and self.edge_dropout > 0:
            mask = torch.rand(ei.size(1)) > self.edge_dropout
            ei = ei[:, mask]
            ea = ea[mask] if ea is not None else None
            
        if self.edge_mode == "weight":
            ea = self.edge_emb(ea).squeeze(-1)
        elif self.edge_mode == "attr":
            ea = self.edge_emb(ea)

        if hasattr(self,'return_all_hiddens') and self.return_all_hiddens:
            hidden_states = [x]

        for idx, (conv, bn) in enumerate(zip(self.convs, self.bns)):

            if self.edge_mode == "update":
                ea = self.edge_updaters[idx](x, ei, ea)
                ea = self.edge_bn[idx](ea)

            if self.edge_mode == "weight":
                h = conv(x, ei, edge_weight=ea)
            else:
                h = conv(x, ei, edge_attr=ea)

            if self.residual:
                scale = self.alpha[idx] if self.gated else 1.0
                h = h * scale + x

            h = bn(h)
            if hasattr(self, "films") and self.films is not None:
                ctx = h[_GLOBAL_ID::65]  # board‑level context rows
                h = self.films[idx](h, batch, ctx)

            if hasattr(self,'act_fn'):
                h = self.act_fn(h)
            else:
                h = F.relu(h)
            x = self.drop(h)

            if self.return_all_hiddens:
                hidden_states.append(x)

        if hasattr(self, "return_for_cnn") and self.return_for_cnn:
            batch_size = int(batch.max().item()) + 1
            return x.view(batch_size, _N_NODES, x.size(1))


        if hasattr(self, "return_before_pool") and self.return_before_pool:
            batch_size = int(batch.max().item()) + 1
            if self.return_all_hiddens:
                flat_states = [h.view(batch_size, -1) for h in hidden_states]
                return torch.stack(flat_states, dim=1)
            return x.view(batch_size, -1)



        if hasattr(self, "MLPFeaturePool") and self.MLPFeaturePool:
            batch_size = int(batch.max().item()) + 1
            xg = x.view(batch_size, _N_NODES, self.hidden_dim)
            feature_pooled = torch.einsum('bfn,nf->bn', xg, self.feature_mlp_weight) + self.feature_mlp_bias
        else:
            feature_pooled = None

        if hasattr(self, "return_all_hiddens") and self.return_all_hiddens:
            pooled = [torch.cat([global_mean_pool(h, batch),
                                 global_max_pool(h, batch)], dim=1)
                      for h in hidden_states]
            g = torch.stack(pooled, dim=1)
        elif hasattr(self, "NodeFdim") and self.NodeFdim is not None:
            batch_size = int(batch.max().item()) + 1
            xg = x.view(batch_size, _N_NODES, self.hidden_dim)
            node_out = xg[:,:,:self.NodeFdim].reshape(batch_size, -1)

            xg_remain = xg[:, :, self.NodeFdim:].reshape(batch_size * _N_NODES, -1)
            mean_out = global_mean_pool(xg_remain, batch)
            max_out = global_max_pool(xg_remain, batch)
            g = torch.cat([node_out, mean_out, max_out], dim=1)
        elif hasattr(self, "NodePool") and self.NodePool:
            batch_size = int(batch.max().item()) + 1
            node_out = self.node_mlp(x).reshape(batch_size, -1)
            feature_pooled = feature_pooled if feature_pooled is not None else torch.cat([global_mean_pool(x, batch),global_max_pool(x, batch)], dim=1)
            g = torch.cat([node_out,
                feature_pooled
            ], dim=1)
        else:
            g = feature_pooled if feature_pooled is not None else torch.cat([global_mean_pool(x, batch),global_max_pool(x, batch)], dim=1)
        return g



def make_encoder(conv_cls, **kw):
    return BaseEncoder(conv_cls, **kw)


EDGE_MODEL_REGISTRY = {
    "GATv2" : lambda **kw: make_encoder(GATv2Conv, **kw),
    "Transf": lambda **kw: make_encoder(TransformerConv, **kw),

    "GCN": lambda **kw: make_encoder(GCNConv, **kw),
    "Cheb": lambda **kw: make_encoder(ChebConv, **kw),
    "Graph": lambda **kw: make_encoder(GraphConv, **kw),

    "GINE"  : lambda **kw: make_encoder(GINEConv,  **kw),
    "GEN"   : lambda **kw: make_encoder(GENConv,   **kw),
    "ResGateGraph": lambda **kw: make_encoder(ResGatedGraphConv, **kw),

    "CustomV1": lambda **kw: make_encoder(CustomV1, **kw),
    "CustomV2": lambda **kw: make_encoder(CustomV2, **kw),
}


def get_GNN_model(name: str, **kwargs):
    return EDGE_MODEL_REGISTRY[name](
        in_feats=_F_DIM,
        edge_in_feats=_EDGE_DIM,
        **kwargs
    )