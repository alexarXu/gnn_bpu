import pandas as pd
import torch
import os
from src.model.GNN_net import GNN_ATTENTION_MODELS, EDGE_MODEL_REGISTRY
from src.model.RNN_net import RNN_MODELS

def get_input_output_list():
    info_df = pd.read_pickle('droso_data/neuron_category.pkl')
    sensory_df = info_df[info_df['Category'] == 'sensory']
    input_list = ['all', 'ascending', 'sensory'] + sorted(list(set(sensory_df['sub_Category'].to_list())))
    output_list = ['all', 'output', 'DN-SEZ', 'DN-VNC', 'RGN']
    return input_list, output_list

def get_ct(name):
    info_df = pd.read_pickle('droso_data/neuron_category.pkl')
    if name == 'all':
        return len(info_df)
    elif name == 'output':
        output_list = {'DN-SEZ', 'DN-VNC', 'RGN'}
        filterd = info_df[info_df['Category'].isin(output_list)]
        return len(filterd)
    else:
        filterd = info_df[info_df['Category'] == name]
        if filterd.empty:
            filterd = info_df[info_df['sub_Category'] == name]
        return len(filterd)

def get_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA device.")

    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Apple MPS device.")

    else:
        device = torch.device("cpu")
        print("Using CPU device.")

    return device


def get_exp_name(config: dict):
    if config.get('GNN_only',False):
        exp_id = 'GNN'
        expanded = config.get('expanded', None)
        if expanded:
            exp_id = f'DPU{expanded}x_' + exp_id
    elif config.get('GCNN', False):
        exp_id = 'GCNN'
    else:
        if config['init'] == 'droso':
            exp_id = 'DPU'
        elif config['init'] == 'droso-propo':
            exp_id = 'DPU_Propo'
            assert config['learnable'] == False
        else:
            raise NotImplementedError("Only support Droso init now")

        expanded = config.get('expanded',None)
        if expanded:
            exp_id = exp_id + f'{expanded}x'

        if config['learnable']:
            exp_id = exp_id + '_Learnable_' + config['learnable_type']
        else:
            exp_id = exp_id + '_Unlearnable'

    maches = set(EDGE_MODEL_REGISTRY.keys()) & config['model_choice'].keys()
    if maches:
        GNN_model = list(maches)[0]
        GNN_config = config['model_choice'][GNN_model]
        if GNN_config.get('residual', False):
            prefix = 'Res'
            if GNN_config.get('gated', False):
                prefix = 'Ga' + prefix
        else:
            prefix = ''

        if config.get('multi_input',False):
            prefix = 'Multi' + prefix
        if GNN_config.get('update_edge',False):
            prefix = 'Edge' + prefix
        if GNN_config.get('use_film', False):
            prefix = 'Film' + prefix
        NodePool = GNN_config.get('NodePool', None)
        if NodePool is not None:
            prefix = 'NodePool' + str(NodePool) + prefix
        NodeFdim = GNN_config.get('NodeFdim', None)
        if NodeFdim is not None:
            prefix = 'NodeFdim' + str(NodeFdim) + prefix
        if GNN_config.get('MLPFeaturePool', False):
            prefix = 'MLPFeaturePool_' + prefix
        if GNN_config.get('use_pretrained', None) is not None:
            prefix = 'Fixed_' + prefix

        act = GNN_config.get('act', None)
        if act:
            prefix = act + prefix


        exp_id = exp_id + f'_{prefix}{GNN_model}' + f"_{GNN_config['hidden_dim']}HiddenDim" + f"_{GNN_config['num_layer']}Layer"
        if GNN_model in GNN_ATTENTION_MODELS:
            exp_id = exp_id + f"_{GNN_config['heads']}Head"
        else:
            num_head = config.get('multi_gnn', None)
            if num_head is not None:
                exp_id = exp_id + f"_{num_head}Head"

    maches = set(RNN_MODELS) & config['model_choice'].keys()
    if maches:
        RNN_model = list(maches)[0]
        exp_id = exp_id + f'_{RNN_model}'
        if config['model_choice'][RNN_model]['residual']:
            exp_id = exp_id + '_Residual'

    if not (config.get('GNN_only', False) or config.get('GCNN', False)):
        exp_id = exp_id + f"_{str(config['timesteps'])}timesteps"

    if config.get('ensemble', False):
        exp_id = exp_id + f"_{str(config.get('ensemble'))}Ensembles"

    exp_id = exp_id + f"_{str(config['train_num_sample'])}"
    exp_id = exp_id + f"_seed{str(config['seed'])}"

    numeric_input = config.get('numeric_input', False)
    if numeric_input:
        exp_id = 'NUM_' + exp_id

    connectome_type = config.get('connectome_type', '1layer')
    if connectome_type == '2layer':
        exp_id = exp_id.replace('NUM_', 'NUM_Conn2L_') if exp_id.startswith('NUM_') else 'Conn2L_' + exp_id

    init_type = config.get('init_type', 'droso')
    if init_type == 'random':
        exp_id = exp_id.replace('NUM_', 'NUM_RandMask_') if exp_id.startswith('NUM_') else 'RandMask_' + exp_id
    elif init_type == 'random_sparsity':
        exp_id = exp_id.replace('NUM_', 'NUM_RandSp_') if exp_id.startswith('NUM_') else 'RandSp_' + exp_id
    elif init_type == 'random_sparsity_rank':
        exp_id = exp_id.replace('NUM_', 'NUM_RandSpRank_') if exp_id.startswith('NUM_') else 'RandSpRank_' + exp_id
    elif init_type == 'random_sparsity_degree':
        exp_id = exp_id.replace('NUM_', 'NUM_RandSpDeg_') if exp_id.startswith('NUM_') else 'RandSpDeg_' + exp_id

    if config.get('match_connectome_weight_range'):
        exp_id = exp_id.replace('NUM_', 'NUM_MatchRng_') if exp_id.startswith('NUM_') else 'MatchRng_' + exp_id
    elif config.get('custom_weight_scale') is not None:
        scale = config.get('custom_weight_scale')
        tag = f'Scale{str(scale).replace(".", "p")}_'
        exp_id = exp_id.replace('NUM_', f'NUM_{tag}') if exp_id.startswith('NUM_') else tag + exp_id

    return exp_id




def count_params(model):
    """Return the number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_out_path(config):
    return os.path.join(config['result_path'], config['exp_id'])
