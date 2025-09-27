import glob
import os
import re
from pathlib import Path

import torch
from tqdm import tqdm
import pickle as pkl
import numpy as np
from .scaler import MinMaxScaler
from torch.utils.data import DataLoader

def increment_path(path, exist_ok=False, sep="", mkdir=False):
    # Increment file or directory path, i.e. runs/exp --> runs/exp{sep}2, runs/exp{sep}3, ... etc.
    path = Path(path)  # os-agnostic
    if path.exists() and not exist_ok:
        suffix = path.suffix
        path = path.with_suffix("")
        dirs = glob.glob(f"{path}{sep}*")  # similar paths
        matches = [re.search(rf"%s{sep}(\d+)" % path.stem, d) for d in dirs]
        i = [int(m.groups()[0]) for m in matches if m]  # indices
        n = max(i) + 1 if i else 2  # increment number
        path = Path(f"{path}{sep}{n}{suffix}")  # update path
    dir = path if path.suffix == "" else path.parent  # directory
    if not dir.exists() and mkdir:
        dir.mkdir(parents=True, exist_ok=True)  # make directory
    return path


class Normalizer:
    def __init__(self, data):
        flat = data.reshape(-1, data.shape[-1])
        self.scaler = MinMaxScaler((-1, 1), clip=True)
        self.scaler.fit(flat)

    def normalize(self, x):
        return x
        batch, seq, ch = x.shape
        x = x.reshape(-1, ch)
        return self.scaler.transform(x).reshape((batch, seq, ch))

    def unnormalize(self, x):
        return x
        batch, seq, ch = x.shape
        x = x.reshape(-1, ch)
        x = torch.clip(x, -1, 1)  # clip to force compatibility
        return self.scaler.inverse_transform(x).reshape((batch, seq, ch))


def vectorize_many(data):
    # given a list of batch x seqlen x joints? x channels, flatten all to batch x seqlen x -1, concatenate
    seq_len = data.shape[1]

    out = [x.reshape(seq_len, -1).contiguous() for x in data]

    global_pose_vec_gt = torch.cat(out, dim=0)
    return global_pose_vec_gt


def get_dataset_loader(batch_size, split='train', num_workers = 8, shuffle=True, full_length=False):
    dataset = DD100lfAll2(split=split, full_length=full_length)
    # collate = None

    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle,
        num_workers = num_workers,
        drop_last = False,
        # collate_fn=collate,
        pin_memory = True,
        persistent_workers=True if num_workers > 0 else False, #https://github.com/Lightning-AI/lightning/issues/10389
    )
    return loader

def to_torch(ndarray):
    if type(ndarray).__module__ == 'numpy':
        return torch.from_numpy(ndarray)
    elif not torch.is_tensor(ndarray):
        raise ValueError("Cannot convert {} to torch tensor".format(
            type(ndarray)))
    return ndarray

def preprocess_data(dataset, split='train'):
        data_path = dataset.datapath
        if split == 'train':
            seq_id_list = dataset._train_id_list
        elif split == 'test':
            seq_id_list = dataset._test_id_list

        all_pose_vec_input = []
        for seq_name in tqdm(seq_id_list):
            group_motion_data = dataset._load_motion_sequence(seq_name)
            seq_len = group_motion_data['group_poses'].shape[1]
            group_pose_vec = dataset._process_poses(group_motion_data, frame_ix = np.arange(seq_len)) # (n_persons, n_frames, 70)
            all_pose_vec_input.append(group_pose_vec.reshape(-1, group_pose_vec.shape[-1])) #(-1, 70)
        all_pose_vec_input = torch.cat(all_pose_vec_input) # (N_all, 70)
        print("Loaded all pose data with shape: ", all_pose_vec_input.shape)
        normalizer = ZNormalizer(all_pose_vec_input)
        print("Saving normalizer to normalizer.pkl")
        pkl.dump(normalizer, open(os.path.join(data_path, "normalizer.pkl"), "wb"))
        
class ZNormalizer:
    def __init__(self, data, eps=1e-10):
        flat = data.reshape(-1, data.shape[-1])
        self.mean = flat.mean(dim=0)
        self.std = flat.std(dim=0)
        self.eps = eps

    def normalize(self, x):
        mean = self.mean.to(x.device)
        std = self.std.to(x.device)
        return ((x - mean)/(std + self.eps))

    def unnormalize(self, x):

        mean = self.mean.to(x.device)
        std = self.std.to(x.device)
        return x * std + mean