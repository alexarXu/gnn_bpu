from torch.utils.data import Dataset
from torch.utils.data import IterableDataset
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from tqdm import tqdm
import glob
import os
import torch
import numpy as np

from src import chess_config as config_lib
from src.chess_utils import bagz
from src import constants
from src.chess_utils.tokenizer import build_tokenizer


TOTAL_RECORD_CT = 530310443


def save_dataset_to_bag(dataset, output_bag_path):
    os.makedirs(os.path.dirname(output_bag_path), exist_ok=True)
    with bagz.BagWriter(output_bag_path) as writer:
        for idx in dataset.indices:
            raw_record = dataset.data_source[idx]
            writer.write(raw_record)
    print(f"Saved {len(dataset.indices)} records to {output_bag_path}")

import os
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool

def decode_record(idx_and_source):
    idx, data_source, coder = idx_and_source
    raw = data_source[idx]
    return coder.decode(raw)

def save_dataset_to_npy(dataset, output_npy_path, n_procs=80, chunk_size=128):
    os.makedirs(os.path.dirname(output_npy_path), exist_ok=True)
    coder       = dataset.coder
    data_source = dataset.data_source
    indices     = dataset.indices

    # prepare the iterable of args for each worker
    args = ((i, data_source, coder) for i in indices)

    pairs = []
    with Pool(processes=n_procs) as pool:
        # imap yields results in-order, with less memory overhead than map
        for fen, win_prob in tqdm(pool.imap(decode_record, args, chunksize=chunk_size),
                                  total=len(indices),
                                  desc="Decoding → .npy"):
            pairs.append((fen, win_prob))

    np.save(output_npy_path, np.array(pairs, dtype=object))
    print(f"Saved {len(pairs)} (fen, win_prob) pairs to {output_npy_path}")

def transform_to_data(x, edge_idx,edge_attr):
    x = torch.tensor(x, dtype=torch.float)  # [N, 34]
    ei = torch.tensor(edge_idx, dtype=torch.long)  # [2, E]
    ea = torch.from_numpy(edge_attr).float()
    return Data(x=x, edge_index=ei, edge_attr=ea)

class BaseChessTransform:
    def __init__(self, config):
        self.config = config

class ConvertStateValueDataToSequence(BaseChessTransform):
    """Converts (fen, win_prob) into a sequence of tokens [S; R]."""
    def __init__(self, config):
        super().__init__(config)
        self.token_fn = build_tokenizer(self.config)

    def __call__(self, fen: str, win_prob: float):
        x, edge_idx, edge_attr = self.token_fn(fen)
        return x, edge_idx, edge_attr, np.array([win_prob,1-win_prob])


_TRANSFORMATION_BY_POLICY = {
    'state_value': ConvertStateValueDataToSequence,
}

class ChessDataset(Dataset):
    def __init__(self,
                 data_path: str,
                 coder_name: str,
                 transform_obj: BaseChessTransform,
                 num_records: int = None,
                 seed: int = 12345,
                 sampling = False):
        super().__init__()
        self.data_source = bagz.BagDataSource(data_path)
        self.coder = constants.CODERS[coder_name]
        self.transform_obj = transform_obj

        total_records = len(self.data_source)
        if num_records is not None and num_records < total_records:
            rng = np.random.default_rng(seed)
            self.indices = rng.choice(total_records, size=num_records, replace=False)
            self.num_records = num_records
        else:
            self.indices = np.arange(total_records)
            self.num_records = total_records

    def __len__(self):
        return self.num_records

    def __getitem__(self, idx: int):
        real_idx = idx if self.indices is None else int(self.indices[idx])
        raw_bytes = self.data_source[real_idx]
        decoded = self.coder.decode(raw_bytes)

        x, edge_idx,edge_attr, win_prob = self.transform_obj(*decoded)
        
        return transform_to_data(x, edge_idx, edge_attr), torch.as_tensor(win_prob, dtype=torch.float32)

def seed_worker(worker_id):
    torch.set_num_threads(1) 
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)

class FullNpyChessIterable(IterableDataset):
    def __init__(self, shard_paths: list[str], transform_obj):
        self.shard_paths = shard_paths
        self.transform   = transform_obj

    def _iter_shards(self, paths_subset):
        for npy_path in paths_subset:
            pairs = np.load(npy_path, allow_pickle=True)
            for fen, wp in pairs:
                x, ei, ea, wp_arr = self.transform(fen, float(wp))
                data  = transform_to_data(x, ei, ea)
                label = torch.as_tensor(wp_arr, dtype=torch.float32)
                yield data, label

    def __iter__(self):
        info = torch.utils.data.get_worker_info()
        if info is None:                       # single-process dataloader
            return self._iter_shards(self.shard_paths)
        # stride through the list so workers get disjoint shards
        return self._iter_shards(self.shard_paths[info.id::info.num_workers])


class NpyChessDataset(Dataset):
    def __init__(self, npy_path, transform_obj):
        self.pairs       = np.load(npy_path, allow_pickle=True)
        self.transform  = transform_obj

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        fen, wp = self.pairs[idx]
        x, edge_idx, edge_attr, win_prob_arr = self.transform(fen, float(wp))
        data  = transform_to_data(x, edge_idx, edge_attr)
        label = torch.as_tensor(win_prob_arr, dtype=torch.float32)
        return data, label

def build_data_loader(config: config_lib.DataConfig,exp_config: dict) -> DataLoader:
    assert exp_config['data_choice'] == 'chess_SV'
    policy_name = 'state_value'
    transform_cls = _TRANSFORMATION_BY_POLICY[policy_name]
    transform_obj = transform_cls(exp_config)
    
    generator = torch.Generator()
    generator.manual_seed(config.seed)

    chess_root = exp_config.get(
        'chess_data_root',
        os.path.join(os.getcwd(), 'data/chess_data'),
    )
    data_path = os.path.join(chess_root, config.split, f'{policy_name}_data.bag')
    if not exp_config.get('sampling', False): # ONLY USING NPY SUBSAMPLEs
        if config.split == 'train':
            if config.num_records >= TOTAL_RECORD_CT//10:
                # Allow training on any device; shard loading does not require CUDA.
                # assert torch.cuda.is_available()

                if config.num_records == TOTAL_RECORD_CT:
                    base_dir = os.path.join(chess_root, 'full_train')
                elif config.num_records == TOTAL_RECORD_CT//10:
                    base_dir = os.path.join(chess_root, '10p_train')
                else:
                    raise ValueError(f"Invalid number of records: {config.num_records}")
                
                # Prefer sharded .npy files when available.
                shard_glob = os.path.join(base_dir, 'shuffle*', 'part*.npy')
                shard_paths = sorted(glob.glob(shard_glob))
                print(f"Found {len(shard_paths)} .npy shard files under {base_dir}")

                if shard_paths:
                    print("Using shard dataset .npy")
                    dataset = FullNpyChessIterable(shard_paths, transform_obj)
                    loader = DataLoader(
                        dataset,
                        batch_size=config.batch_size,
                        num_workers=20,
                        persistent_workers=True,
                        worker_init_fn=seed_worker,
                        prefetch_factor=2,
                        pin_memory=False,
                        generator=generator
                    )
                    return loader
                else:  # Fall back to a subsampled .npy dataset.
                    print(f"No shards found, falling back to subsampling")
                    npy_subsample_data_path = os.path.join(
                        chess_root,
                        'subsample_train',
                        f'subsample_{config.num_records}_seed{config.seed}.npy',
                    )
                    if os.path.exists(npy_subsample_data_path):
                        print(f"Using existing subsampled npy dataset\n{npy_subsample_data_path}")
                        data_path = npy_subsample_data_path
                    else:  # Create a new subsample .npy file.
                        dataset = ChessDataset(
                            data_path=data_path,
                            coder_name=policy_name,
                            transform_obj=transform_obj,
                            num_records=config.num_records,
                            seed=config.seed,
                            sampling=True
                        )
                        num_records = len(dataset)
                        print(f"Creating subsample .npy dataset with {num_records} records.")
                        output_npy = os.path.join('data', 'chess_data', 'subsample_train', 
                                                f'subsample_{num_records}_seed{config.seed}.npy')
                        save_dataset_to_npy(dataset, output_npy)
                        data_path = output_npy


    if data_path.endswith('.npy'):
        dataset = NpyChessDataset(data_path, transform_obj)
    else: # ONLY used in subsampling now
        assert (exp_config.get('sampling', False) or config.split == 'test')
        dataset = ChessDataset(
            data_path=data_path,
            coder_name=policy_name,
            transform_obj=transform_obj,
            num_records=config.num_records,
            seed=config.seed,
        )

    if torch.backends.mps.is_available():
        loader = DataLoader(
            dataset,
            batch_size=config.batch_size,
            shuffle=config.shuffle,
            num_workers=20,
            persistent_workers=True,
            worker_init_fn=seed_worker,
            prefetch_factor=2,
            pin_memory=False,
            generator=generator
        )
    else:
        loader = DataLoader(
            dataset,
            batch_size=config.batch_size,
            shuffle=config.shuffle,
            num_workers=20,
            persistent_workers=True,
            worker_init_fn=seed_worker,
            prefetch_factor=2,
            pin_memory=False,
            generator=generator
        )

    return loader
