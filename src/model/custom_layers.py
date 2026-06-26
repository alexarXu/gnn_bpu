import torch
from torch import Tensor
from torch.nn import Linear
from typing import Union
from torch_geometric.nn import MessagePassing
from torch_geometric.typing import Adj, OptPairTensor, OptTensor, Size
from torch_geometric.nn.inits import reset

import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import numpy as np

class SparseLinear(nn.Module):
    def __init__(
            self,
            in_features: int,
            out_features: int,
            sparse_prob_type = None,
            init_type = None,
            trainable: bool = False,
            conn = None,
            refactor_scale=100.0
    ):
        super().__init__()

        if init_type == 'random': # random init
            sparsity = 0.99
            if sparse_prob_type == 'input_output':
                sparsity = 1.0 - 882.0 / 400.0 / 430.0
            elif sparse_prob_type is not None:
                raise NotImplementedError('sparsity type not implemented')

            self.linear = nn.Linear(in_features, out_features, bias=False)
            prune.random_unstructured(self.linear, name='weight', amount=sparsity)
            prune.remove(self.linear, 'weight')

            with torch.no_grad():
                self.linear.weight.data.mul_(refactor_scale)

            nonzeros = torch.count_nonzero(self.linear.weight).item()
            print(f"Random init [SparseLinear] non-zero weights: {nonzeros}")

        else:
            init = conn['W']
            if sparse_prob_type == 'input_output':
                temp = init[conn['sensory_idx'], :]
                weight = temp[:, conn['output_idx']]
            elif sparse_prob_type == 'all_all':
                weight = init
            else:
                raise NotImplementedError('invalid sparsity type')

            if init_type == 'droso':
                weight = torch.from_numpy(weight).float()
                weight = weight.t().contiguous()  # [out_f, in_f]
            elif init_type == 'droso_permute':
                temp = weight.flatten()
                temp = np.random.permutation(temp)
                temp = temp.reshape(weight.shape)
                temp = torch.from_numpy(temp).float()
                weight = temp.t().contiguous()  # [out_f, in_f]

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
    


