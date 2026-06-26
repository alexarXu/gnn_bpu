import os

import pandas as pd
import numpy as np

from src.droso_matrix.utils import normalize_matrix

CONN_TYPE_LIST = ['aa','ad','da','dd']


def load_drosophila_matrix(csv_path, signed=False):
    """
    Load and process a Drosophila connectivity matrix.
    """
    W_df = pd.read_csv(csv_path, index_col=0, header=0)
    W = W_df.values.astype(np.float32)

    # Normalize depending on whether it's signed or unsigned
    if signed:
        max_abs = np.max(np.abs(W))
        W_norm = W / max_abs if max_abs != 0 else W
    else:
        W_min, W_max = W.min(), W.max()
        W_norm = (W - W_min) / (W_max - W_min + 1e-8)

    return W_norm

def load_connectivity_data(connectivity_path, annotation_path, rescale_factor=4e-2, normalization=None):
    """
    Load and preprocess connectivity matrix and annotation data for Drosophila.
    """
    df_annot = pd.read_csv(annotation_path)

    mask = (df_annot['celltype'] == 'sensory') & (df_annot['additional_annotations'] == 'visual')
    sensory_visual_ids = []
    for _, row in df_annot[mask].iterrows():
        for col in ['left_id', 'right_id']:
            id_str = str(row[col]).lower()
            if id_str != "no pair":
                sensory_visual_ids.append(int(id_str))

    sensory_visual_ids = sorted(set(sensory_visual_ids))
    print(f"Found {len(sensory_visual_ids)} sensory-visual neuron IDs")

    df_conn = pd.read_csv(connectivity_path, index_col=0)
    df_conn.index = df_conn.index.astype(int)
    df_conn.columns = df_conn.columns.astype(int)

    valid_sensory_ids = [nid for nid in sensory_visual_ids if nid in df_conn.index]
    other_ids = [nid for nid in df_conn.index if nid not in valid_sensory_ids]

    df_reindexed = df_conn.loc[valid_sensory_ids + other_ids, valid_sensory_ids + other_ids]

    adj_matrix = df_reindexed.values
    adj_matrix = normalize_matrix(adj_matrix, mode=normalization)
    adj_matrix = adj_matrix * rescale_factor

    num_S = len(valid_sensory_ids)
    return {
        'W': adj_matrix,
        'W_ss': adj_matrix[:num_S, :num_S],
        'W_sr': adj_matrix[:num_S, num_S:],
        'W_rs': adj_matrix[num_S:, :num_S],
        'W_rr': adj_matrix[num_S:, num_S:],
        'sensory_ids': valid_sensory_ids
    }


def load_sio_conn(connectivity_path, annotation_path, rescale_factor=4e-2, normalization='minmax',
                      input_type='visual',
                      output_type = 'output'):


    output_types = {'DN-SEZ', 'DN-VNC', 'RGN'}
    if 'npz' in connectivity_path: # Tingshan's expanded matrix
        connnectivity = np.load(connectivity_path)
        W = connnectivity['W_exp']
        z = connnectivity['z_exp']
        sensory_idx = list(np.where(z == 0)[0])
        internal_idx = list(np.where(z == 1)[0])
        output_idx = list(np.where(z == 2)[0])
        ordered_ids = sensory_idx + internal_idx  + output_idx
        W = W[np.ix_(ordered_ids, ordered_ids)]
        W = W * rescale_factor

        return {
            'W': W,  # Now in SIO order
            'sensory_idx': sensory_idx,
            'internal_idx': internal_idx,
            'output_idx': output_idx,
            'input_type': input_type,
            'output_type': output_type,
        }


    df_category = pd.read_pickle(annotation_path)
    if input_type in ['sensory', 'ascending']:
        df_input = df_category[df_category['Category'] == input_type]
    elif input_type != 'all':
        df_input = df_category[df_category['sub_Category'] == input_type]
    else:
        df_input = df_category
    input_ids = df_input['ID'].astype(int).tolist()

    if output_type == 'output':
        df_output = df_category[df_category['Category'].isin(output_types)]
    elif output_type != 'all':
        df_output = df_category[df_category['Category'] == output_type]
    else:
        df_output = df_category
    output_ids = df_output['ID'].astype(int).tolist()

    df_KC = df_category[df_category['Category'] == 'KC']
    KC_ids = df_KC['ID'].astype(int).tolist()

    KC_ids = sorted(set(KC_ids))
    input_ids = sorted(set(input_ids))
    output_ids = sorted(set(output_ids))

    df_conn = pd.read_csv(connectivity_path, index_col=0)
    df_conn.index = df_conn.index.astype(int)
    df_conn.columns = df_conn.columns.astype(int)
    all_neuron_ids = sorted(df_conn.index.tolist())
    print(f"Connectivity matrix contains {len(all_neuron_ids)} neurons")

    valid_sensory_ids = [nid for nid in input_ids if nid in all_neuron_ids]
    valid_output_ids = [nid for nid in output_ids if nid in all_neuron_ids]
    valid_KC_ids = [nid for nid in KC_ids if nid in all_neuron_ids]

    Other_ids = [
        nid for nid in all_neuron_ids
        if nid not in valid_sensory_ids and nid not in valid_output_ids and nid not in valid_KC_ids
    ]

    # Create the ordered adjacency matrix
    ordered_ids = valid_sensory_ids + valid_KC_ids + Other_ids + valid_output_ids
    df_conn_sio = df_conn.loc[ordered_ids, ordered_ids]
    adjacency = df_conn_sio.values  # shape: [N, N]

    # Apply normalization
    adjacency = normalize_matrix(adjacency, mode=normalization)
    adjacency = adjacency * rescale_factor

    return {
        'W': adjacency,  # Now in SIO order
        'sensory_idx': [ordered_ids.index(i) for i in valid_sensory_ids],
        'KC_idx': [ordered_ids.index(i) for i in valid_KC_ids],
        'internal_idx': [ordered_ids.index(i) for i in Other_ids],
        'output_idx': [ordered_ids.index(i) for i in valid_output_ids],
        'input_type': input_type,
        'output_type': output_type,
    }

def get_conn_path(exp_config, cfg_data):
    assert exp_config['signed'] is True
    droso_dir = cfg_data.get('droso_dir', 'droso_data')
    conn_type = exp_config.get('conn_type', 'ad')
    if conn_type != 'all':
        return os.path.join(droso_dir, f'{conn_type}_signed_connectivity_matrix.csv')
    return [os.path.join(droso_dir, f'{t}_signed_connectivity_matrix.csv') for t in CONN_TYPE_LIST]

def load_connectivity_info(exp_config,cfg_data, input_type, output_type, expanded = None, sio=True):
    rescale_factor = cfg_data.get('rescale_factor', 4e-2)
    if expanded is not None:
        droso_dir = cfg_data.get('droso_dir', 'droso_data')
        connectivity_path = os.path.join(droso_dir, 'expanded_droso', f'expansion_{expanded}x.npz')
        rescale_factor =  0.01
    else:
        connectivity_path = get_conn_path(exp_config, cfg_data)
    print(cfg_data.get('rescale_factor', 4e-2))
    return load_sio_conn(
        connectivity_path=connectivity_path,
        annotation_path=cfg_data["annotation_path"],
        rescale_factor=rescale_factor,
        normalization=cfg_data.get('normalization', None),
        input_type=input_type,
        output_type=output_type,
    )


def load_connectivity_info_2states(
            exp_config,
            cfg_data,
            input_type,
            output_type,
        ):
    connectivity_path_list = get_conn_path(exp_config,cfg_data)
    sensory,KC,internal,output = None,None,None,None
    for conn_path in connectivity_path_list:
        conn_temp = load_sio_conn(
            connectivity_path=conn_path,
            annotation_path=cfg_data["annotation_path"],
            rescale_factor=cfg_data.get('rescale_factor', 4e-2),
            normalization=cfg_data.get('normalization', None),
            input_type=input_type,
            output_type=output_type,
        )
        name = conn_path.split('/')[-1].split('_', 1)[0]
        if sensory is None:
            sensory = conn_temp['sensory_idx']
            KC = conn_temp['KC_idx']
            internal = conn_temp['internal_idx']
            output = conn_temp['output_idx']
            conn = conn_temp
            conn[f'W_{name}'] = conn.pop(f'W')
        else:
            assert sensory == conn_temp['sensory_idx']
            assert KC == conn_temp['KC_idx']
            assert internal == conn_temp['internal_idx']
            assert output == conn_temp['output_idx']
            conn[f'W_{name}'] = conn_temp['W']
    return conn