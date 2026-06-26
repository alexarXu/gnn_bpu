import os
from torch_geometric.nn import global_mean_pool

from src.droso_matrix.connectome import load_connectivity_info
from src.droso_matrix.utils import get_weight_matrix
from torch_geometric.data import Data

from src.model.CNN_net import CNN_MODEL_REGISTRY, get_CNN_model
from src.model.GNN_net import EDGE_MODEL_REGISTRY, get_GNN_model
from src.model.RNN_net import RNN_MODELS, BaseRNN
import torch
import torch.nn as nn
from src.model.custom_layers import SparseLinear
from src.chess_utils.tokenizer import N_NODES as NUM_NODE


class GNN_Pool(nn.Module):
    def __init__(self):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.gmp = nn.AdaptiveMaxPool1d(1)

    def forward(self, x):
        x_t = x.permute(0, 2, 1)
        avg_pool = self.gap(x_t)
        max_pool = self.gmp(x_t)
        pooled = torch.cat([avg_pool, max_pool], dim=1).squeeze()
        return pooled

class GCNN_Head(nn.Module):
    def __init__(self, conn,
                 gnn_: nn.Module,
                 cnn_: nn.Module,
                 # fc_input_dim,
                 trainable_sparse = False,
                 sparse_type = None,
                 init_type = False,
                 dropout_rate=0.2,
                 sparse_dim = 512,
                 pre_trained_gnn = False):
        super().__init__()
        self.gnn = gnn_
        if pre_trained_gnn:
            self.gnn.return_for_cnn = True
            for param in self.gnn.parameters():
                param.requires_grad = False

        self.cnn = cnn_
        fc_input_dim = cnn_.feat_dim + gnn_.hidden_dim * 2
        if sparse_type == 'input_output':
            self.input_fc = nn.Linear(fc_input_dim, len(conn['sensory_idx']))
            self.sparse_fc = SparseLinear(len(conn['sensory_idx']), len(conn['output_idx']), sparse_prob_type=sparse_type,init_type = init_type, trainable=trainable_sparse,conn = conn)
            self.out_fc = nn.Linear(len(conn['output_idx']), 2)
        elif sparse_type == 'all_all':
            self.input_fc = nn.Linear(fc_input_dim, 2952)
            self.sparse_fc = SparseLinear(2952, 2952, sparse_prob_type=sparse_type,init_type = init_type, trainable=trainable_sparse,conn = conn)
            self.out_fc = nn.Linear(2952, 2)
        else:
            self.input_fc = nn.Linear(fc_input_dim, sparse_dim)
            self.sparse_fc = SparseLinear(sparse_dim, sparse_dim,sparse_prob_type = sparse_type,init_type = init_type,trainable = trainable_sparse, conn = conn)
            self.out_fc = nn.Linear(sparse_dim, 2)
        self.dropout = nn.Dropout(dropout_rate)
        self.gnn_pool = GNN_Pool()


    def forward(self, data: Data) -> torch.Tensor:
        gnn_out = self.gnn(data)
        gnn_feat = self.gnn_pool(gnn_out)

        cnn_in = gnn_out[:,:-1,:]# chop off global dim for CNN input
        cnn_in = cnn_in.reshape(cnn_in.size(0), 8, 8, cnn_in.size(2)) # reshape back to 8*8
        cnn_feat = self.cnn(cnn_in) # B, cnn_.feat_dim

        mlp_in = torch.cat([gnn_feat, cnn_feat], dim=1)
        x = self.input_fc(mlp_in)
        x = self.dropout(x)
        x = nn.functional.relu(x)
        x = self.sparse_fc(x)
        x = nn.functional.relu(x)
        x = self.out_fc(x)

        return x






class GNN_Head(nn.Module):
    def __init__(self, conn,
                 gnn_: nn.Module,fc_input_dim,trainable_sparse = False,
                 sparse_type = None,
                 init_type = False,
                 dropout_rate=0.2,
                 sparse_dim = 512):
        super().__init__()
        self.gnn = gnn_
        if sparse_type == 'input_output':
            self.input_fc = nn.Linear(fc_input_dim, len(conn['sensory_idx']))
            self.sparse_fc = SparseLinear(len(conn['sensory_idx']), len(conn['output_idx']), sparse_prob_type=sparse_type,init_type = init_type, trainable=trainable_sparse,conn = conn)
            self.out_fc = nn.Linear(len(conn['output_idx']), 2)
        elif sparse_type == 'all_all':
            self.input_fc = nn.Linear(fc_input_dim, 2952)
            self.sparse_fc = SparseLinear(2952, 2952, sparse_prob_type=sparse_type,init_type = init_type, trainable=trainable_sparse,conn = conn)
            self.out_fc = nn.Linear(2952, 2)
        else:
            self.input_fc = nn.Linear(fc_input_dim, sparse_dim)
            self.sparse_fc = SparseLinear(sparse_dim, sparse_dim,sparse_prob_type = sparse_type,init_type = init_type,trainable = trainable_sparse, conn = conn)
            self.out_fc = nn.Linear(sparse_dim, 2)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, data: Data) -> torch.Tensor:
        if isinstance(self.gnn, nn.ModuleList):
            outs = [g(data) for g in self.gnn]
            gnn_out = torch.cat(outs, dim=1)
        else:
            gnn_out = self.gnn(data)

        if gnn_out.ndim != 2:
            gnn_out = gnn_out.reshape(gnn_out.size(0), -1)

        x = self.input_fc(gnn_out)
        x = self.dropout(x)
        x = nn.functional.relu(x)
        x = self.sparse_fc(x)
        x = nn.functional.relu(x)
        x = self.out_fc(x)

        return x


class GNN_RNN(nn.Module):
    def __init__(self, gnn_: nn.Module,rnn_:nn.Module,):
        super().__init__()
        self.gnn = gnn_
        self.rnn = rnn_
        if isinstance(self.rnn, nn.ModuleList):
            output_dim = rnn_[0].output_dim * len(self.rnn)
            self.num_rnn = len(self.rnn)
        else:
            output_dim = rnn_.output_dim
        self.outfc = nn.Linear(output_dim, 2)

    def forward(self, data: Data) -> torch.Tensor:
        if isinstance(self.gnn, nn.ModuleList):
            outs = [g(data) for g in self.gnn]
            gnn_out = torch.cat(outs, dim=1)
        else:
            gnn_out = self.gnn(data)

        if gnn_out.ndim == 2:
            if isinstance(self.rnn, nn.ModuleList):
                outs = [r(gnn_out) for r in self.rnn]
                rnn_out = torch.cat(outs, dim=1)
            else:
                rnn_out = self.rnn(gnn_out)
        else:#(gnn_out.ndim == 3)
            outs = [self.rnn[i](gnn_out[:,i+1,:]) for i in range(self.num_rnn)]
            rnn_out = torch.cat(outs, dim=1)
        return self.outfc(rnn_out)

class Multi_input_GNN_RNN(nn.Module):
    def __init__(self, gnn_: nn.Module,rnn_:nn.Module,):
        super().__init__()
        print("USING MULTI INPUT TO RNN")
        self.gnn = gnn_
        self.rnn = rnn_
        self.outfc = nn.Linear(rnn_.output_dim, 2)

    def forward(self, data: Data) -> torch.Tensor:
        gnn_out = self.gnn(data)
        rnn_out = self.rnn(gnn_out)
        return self.outfc(rnn_out)


def init_rnn(key, config,W_init,conn,exp_config,rnn_input_dim,gnn_layer_ct = None):
    if key == 'BaseRNN':
        base_rnn_init_dict = dict(W_init=W_init,
                        input_dim=rnn_input_dim,
                        sensory_idx=conn['sensory_idx'],
                        KC_idx=conn['KC_idx'],
                        internal_idx=conn['internal_idx'],
                        output_idx=conn['output_idx'],

                        learnable=exp_config.get('learnable'),
                        dropout_rate=exp_config.get('dropout_rate', 0.2),
                        timesteps=exp_config.get('timesteps'),
                        use_residual=config.get('residual'),
                        learnable_type=exp_config.get('learnable_type', None),)

        if exp_config.get('rnn_after_each_layer',False):
            assert gnn_layer_ct is not None
            return nn.ModuleList([
                BaseRNN(**base_rnn_init_dict)
                for _ in range(gnn_layer_ct)
            ])

        # (else)
        num_rnn = config.get('num_rnn',1)
        if num_rnn == 1:
            return BaseRNN(**base_rnn_init_dict)
        else:
            return nn.ModuleList([
                    BaseRNN(**base_rnn_init_dict)
                    for _ in range(num_rnn)
                ])

    else:
        raise ValueError("Invalid RNN model name encountered")


def initialize_model(exp_config):
    assert exp_config['data_choice'] == 'chess_SV'
    GNN_only = exp_config.get('GNN_only', False)
    GCNN = exp_config.get('GCNN', False)
    assert sum([GNN_only, GCNN]) <= 1 # assert GNN only or GCNN or (GNN + RNN)(default)

    droso_config = exp_config['droso_config']
    conn = load_connectivity_info(
        exp_config=exp_config,
        cfg_data=droso_config,
        input_type=exp_config.get('input_type', 'all'),
        output_type=exp_config.get('output_type', 'all'),
        expanded = exp_config.get('expanded', None)
    )
    W_init = get_weight_matrix(conn['W'], exp_config.get('init','droso'))

    model_type = exp_config['model_choice']
    num_gnn = exp_config.get('multi_gnn', None)
    gnn, rnn, gnn_layer_ct = None,None,None
    cnn = None
    rnn_input_dim = 0
    pre_trained_gnn = False
    cnn_in_channel = None
    for key in model_type.keys():
        if model_type[key] is not None:
            if key in EDGE_MODEL_REGISTRY.keys():
                if GCNN: # feed to CNN
                    model_type[key]['return_for_cnn'] = True
                    cnn_in_channel = model_type[key]['hidden_dim']
                else: # returning from GNN
                    return_ct = model_type[key]['hidden_dim'] if model_type[key].get('MLPFeaturePool', False) else (model_type[key]['hidden_dim'] * 2)

                if model_type[key].get('use_pretrained', None) is not None:
                    assert GCNN
                    pre_trained_gnn = True
                    gnn_path = model_type[key]['use_pretrained']
                    model_path = os.path.join(gnn_path,'best_test_model.pth')
                    if torch.backends.mps.is_available():
                        gnn_pretrained = torch.load(model_path, weights_only=False,map_location = torch.device('cpu'))
                    else:
                        gnn_pretrained = torch.load(model_path,weights_only=False)
                    gnn = gnn_pretrained['model'].gnn
                elif num_gnn is not None and num_gnn > 1:# Multi GNN feed into 1 RNN
                    assert not GCNN
                    gnn = nn.ModuleList([
                        get_GNN_model(
                            key,
                            **model_type[key]
                        )
                        for _ in range(num_gnn)
                    ])
                    rnn_input_dim += return_ct * num_gnn
                else:
                    gnn = get_GNN_model(key,
                                        return_all_hiddens = exp_config.get('multi_input', False) or exp_config.get('rnn_after_each_layer',False),
                                        **model_type[key])
                    gnn_layer_ct = model_type[key]['num_layer']
                    if not GCNN:
                        rnn_input_dim += return_ct

                    NodeFdim = model_type[key].get('NodeFdim', None)
                    NodePool = model_type[key].get('NodePool', None)
                    if NodePool is not None:
                        assert not GCNN
                        rnn_input_dim += NodePool * NUM_NODE
                    elif NodeFdim is not None:
                        assert not GCNN
                        rnn_input_dim += NodeFdim * NUM_NODE
            elif key in CNN_MODEL_REGISTRY.keys():
                # Skip CNN construction in GNN-only mode.
                if GNN_only:
                    continue
                    
                # Use a default channel count when GNN output dim is unavailable.
                if cnn_in_channel is None:
                    cnn_in_channel = model_type[key].get('input_channels', 12)
                cnn_args = {'input_channels': cnn_in_channel,
                            'use_time_embedding': False,
                            'time_dim': 50,
                            'time_method': 'material',
                            'time_embedding_type': 'sinusoidal',
                            'use_position_embedding': True,
                            'position_embedding_type': 'simple',
                            'fourier_K': 8,
                            'with_coord': False}
                merged = {**cnn_args, **model_type[key]}
                cnn = get_CNN_model(key,**merged)
    assert gnn is not None

    if not GNN_only and not GCNN:
        for key in model_type.keys():
            if key in RNN_MODELS:
                rnn = init_rnn(key, model_type[key], W_init,conn,exp_config,rnn_input_dim,gnn_layer_ct)
        assert rnn is not None
        if exp_config.get('multi_input', False): #  multi input
            assert isinstance(gnn, nn.ModuleList) == False
            model = Multi_input_GNN_RNN(gnn,rnn)
        else:
            model = GNN_RNN(gnn,rnn)
    elif GCNN:
        assert cnn is not None
        model = GCNN_Head(
            conn,
            gnn,
            cnn,
            # fc_input_dim,
            trainable_sparse=exp_config.get('trainable_sparse', False),
            sparse_type=exp_config.get('sparse_type', None),
            init_type=exp_config.get('init_type', None),
            dropout_rate=exp_config.get('dropout_rate', 0.2), 
            pre_trained_gnn = pre_trained_gnn)
    else: # GNN_only == True
        if exp_config.get('multi_input', False):  # multi input
            fc_input_dim = rnn_input_dim * (gnn_layer_ct + 1)
        else:
            fc_input_dim = rnn_input_dim

        model = GNN_Head(
                    conn,
                    gnn,
                    fc_input_dim,
                    trainable_sparse = exp_config.get('trainable_sparse',False),
                    sparse_type = exp_config.get('sparse_type',None),
                    init_type = exp_config.get('init_type',None),
                    dropout_rate=exp_config.get('dropout_rate', 0.2),)
    if exp_config.get('numeric_input',False):
        model.numeric_input = True
    return model