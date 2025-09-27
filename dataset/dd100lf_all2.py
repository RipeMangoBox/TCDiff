import numpy as np
import torch
from torch.utils.data import Dataset
import os
import random
from dataset.preprocess import Normalizer, vectorize_many

def to_torch(ndarray):
    if type(ndarray).__module__ == 'numpy':
        return torch.from_numpy(ndarray)
    elif not torch.is_tensor(ndarray):
        raise ValueError("Cannot convert {} to torch tensor".format(
            type(ndarray)))
    return ndarray


joint_num = 22

def compute_contacts(dance):
    feet = dance.reshape(-1, 22, 3)[:, (7, 8, 10, 11)]
    feetv = np.zeros(feet.shape[:2]) # bs x seq x feet_joints x 3
    feetv[:, :-1] = np.linalg.norm(feet[:, 1:] - feet[:, :-1], axis=-1) # Compute velocity v for all but the last frame (boundary case excluded)
    contacts = feetv < 0.01  # Compute foot-contact labels
    return contacts

# def compute_contacts(dance):
#     # Assuming dance shape is (num_frames, 22, 3) after reshape
#     # dance shape could also be (batch_size, num_frames, 22, 3), the logic holds
#     feet = dance.reshape(-1, 22, 3)[:, (7, 8, 10, 11)]

#     # Calculate displacement vectors between consecutive frames
#     # np.diff is a cleaner way to do arr[1:] - arr[:-1]
#     velocity_vectors = np.diff(feet, axis=0) # Shape: (T-1, 4, 3)

#     # Calculate the magnitude of velocity for each foot joint
#     velocity_magnitudes = np.linalg.norm(velocity_vectors, axis=-1) # Shape: (T-1, 4)

#     # A velocity close to zero means the foot is in contact with the ground
#     contacts = velocity_magnitudes < 0.01

#     # To match the original sequence length, you can pad the contacts array.
#     # A common practice is to duplicate the first or last frame's contact status.
#     # Here, we'll pad at the end with False (not in contact).
#     padding = np.zeros((1, contacts.shape[1]), dtype=bool)
#     contacts_padded = np.concatenate((contacts, padding), axis=0)

#     return contacts_padded

class DD100lfAll2(Dataset):
    def __init__(self, split='train', full_length=False, normalizer=None):
        self.music_root = "../ReactDance/data_lazy/music"
        self.motion_root = "../ReactDance/data_lazy/motion"
        self.datapath = "./data"
        self.split = split
        self.dances = {'pos3dl':[], 'pos3df':[], 'music':[]}
        dtypes = ['pos3d']
        self.names = []
        
        self.max_length = 150
        self.full_length = full_length
        
        music_files = {}
        agent_files = {'leader':{}, 'follower':{}}

        def _process_files(agent_files, np_music):       
            this_pair = {}
            for agent in agent_files:
                # For each dtype_folder, load the corresponding file
                for dtype_folder in dtypes:
                    dance_path = agent_files[agent][take].replace('pos3d', dtype_folder)
                    if not os.path.isfile(dance_path):
                        continue
                    np_dance = np.load(dance_path)
                    np_dance = np_dance[:len(np_dance) - len(np_dance)%4, :] # to fit encodec down sample stragegy
                    this_pair[(agent, dtype_folder)] = np_dance
                
            for dtype_folder in dtypes:
                if (('leader', dtype_folder) not in this_pair) or (('follower', dtype_folder) not in this_pair):
                    continue
                ldance = this_pair[('leader', dtype_folder)]
                fdance = this_pair[('follower', dtype_folder)]
                seq_len = min(len(ldance), len(fdance))
                ldance = ldance[:seq_len, :joint_num*3]
                fdance = fdance[:seq_len, :joint_num*3]
                
                # Compute foot velocities
                lcontacts = compute_contacts(ldance)
                fcontacts = compute_contacts(fdance)
                ldance = np.concatenate([lcontacts, ldance], axis=-1)
                fdance = np.concatenate([fcontacts, fdance], axis=-1)

                if not self.full_length:
                    stride = 20 #TODO, stride 8?
                    for i in range(0, seq_len - self.max_length, stride):
                        np_dance_sub_seq_l = ldance[i: i + self.max_length]
                        np_dance_sub_seq_f = fdance[i: i + self.max_length]
                        np_music_sub_seq = np_music[i: i + self.max_length]

                        if len(np_dance_sub_seq_l) != self.max_length or len(np_dance_sub_seq_f) != self.max_length or len(np_music_sub_seq) != self.max_length:
                            continue

                        self.dances[dtype_folder+'l'].append(np_dance_sub_seq_l)
                        self.dances[dtype_folder+'f'].append(np_dance_sub_seq_f)
                        
                        # add music only once
                        if dtype_folder == dtypes[0]:
                            self.dances['music'].append(np_music_sub_seq)
                            self.names.append(take)
                else:
                    self.dances[dtype_folder+'l'].append(ldance[:seq_len])
                    self.dances[dtype_folder+'f'].append(fdance[:seq_len])
                    self.dances['music'].append(np_music[:seq_len])
                    self.names.append(take)

        fnames = os.listdir(os.path.join(self.motion_root, 'pos3d', self.split))
        mnames = os.listdir(os.path.join(self.music_root,  'jukebox', self.split))
                
        for mname in mnames:
            path = os.path.join(self.music_root, 'jukebox', self.split, mname)
            music_files[mname[:-4]] = path
            
        for fname in fnames:
            path = os.path.join(self.motion_root, 'pos3d', self.split, fname)
            if path.endswith('_00.npy'):
                agent_files['follower'][fname[:-7]] = path
            elif path.endswith('_01.npy'):
                agent_files['leader'][fname[:-7]] = path

        self._train_id_list = []
        for take in agent_files['follower']:
            if take in agent_files['leader'] and take in music_files:
                # Load to check length for filtering
                np_music = np.load(music_files[take]).astype(np.float32)
                np_music = np_music[:len(np_music) - len(np_music) % 4]
                seq_len = len(np_music)
                if seq_len >= self.max_length or self.full_length:
                    self._train_id_list.append(take)
                    
        for take in agent_files['follower']:
            if take not in agent_files['leader'] or take not in music_files:
                continue
            # music:
            music_path = music_files[take]
            np_music = np.load(music_path).astype(np.float32)
            # For each dtype, process files
            _process_files(agent_files, np_music)
        
        # 将长度对齐操作移到__init__阶段
        # 对self.dances中的所有序列进行长度对齐，裁剪到最小长度的最近的4的倍数
        # 合并为一次循环，使用 (k, v) 方式遍历
        keys = ['pos3dl', 'pos3df', 'music']
        for i in range(len(self.dances['pos3dl'])):
            min_len = min(len(self.dances[k][i]) for k in keys)
            min_len = min_len - min_len % 4
            for k in keys:
                self.dances[k][i] = self.dances[k][i][:min_len]
            
        print('DD100lfAll2 dataset loaded!')   
    
        if self.split == 'train':
            global_pose_vec_input = torch.from_numpy(np.concatenate([self.dances['pos3dl'], self.dances['pos3df']], axis=0)).float().detach()
            global_pose_vec_input = vectorize_many(global_pose_vec_input)
            self.normalizer = Normalizer(global_pose_vec_input)
        elif self.split == 'test':
            assert normalizer is not None
            self.normalizer = normalizer
        # self.dances['pos3dl'] = self.normalizer.normalize(self.dances['pos3dl'])
        # self.dances['pos3df'] = self.normalizer.normalize(self.dances['pos3df'])

    def _load_music_features(self, seq_name):
        music_path = os.path.join(self.music_root, 'jukebox', self.split, f"{seq_name}.npy")
        music_features = np.load(music_path).astype(np.float32)
        return music_features
    
    def _load_motion_sequence(self, seq_name):
        leader_path = os.path.join(self.motion_root, 'pos3d', self.split, f"{seq_name}_01.npy")
        follower_path = os.path.join(self.motion_root, 'pos3d', self.split, f"{seq_name}_00.npy")
        music_features = self._load_music_features(seq_name)
        lmotion = np.load(leader_path)
        fmotion = np.load(follower_path)

        min_len = min(len(lmotion), len(fmotion), len(music_features))
        lmotion = lmotion[:min_len, :66]
        fmotion = fmotion[:min_len, :66]
        # lroot_init = lmotion[0, :3]
        # lmotion = lmotion - np.tile(lroot_init, (min_len, 22))
        # fmotion = fmotion - np.tile(lroot_init, (min_len, 22))
        music_features = music_features[:min_len]
        
        lcontacts = compute_contacts(lmotion)
        fcontacts = compute_contacts(fmotion)
        lmotion = np.concatenate([lcontacts, lmotion], axis=-1)
        fmotion = np.concatenate([fcontacts, fmotion], axis=-1)

        group_motion_data = {
            'group_poses': np.stack([lmotion, fmotion], axis=0),  # (2, seq_len, 66)
            'group_trans': None,  # Not separate in DD100, included in xyz
            'meta': {'orig_start': 0, 'orig_end': min_len}
        }
        
        return group_motion_data
    
    def _process_poses(self, group_motion_data, frame_ix):
        group_poses = group_motion_data['group_poses'][:, frame_ix]  # (2, target_seq_len, 66)
        n_persons, seq_len, pose_dim = group_poses.shape

        # Apply DD100-specific processing
        lmotion = group_poses[0]  # Leader
        fmotion = group_poses[1]  # Follower

        group_poses = np.stack([lmotion, fmotion], axis=0)  # (2, gt_length, 66), note: length may be seq_len-1 due to drop
        group_poses = to_torch(group_poses)

        gt_length = group_poses.shape[1]
        if gt_length < self.max_length:
            padding_len = self.max_length - gt_length
            padding_zeros = torch.zeros((n_persons, padding_len, pose_dim))
            group_poses = torch.cat((group_poses, padding_zeros), dim=1)

        ret = group_poses.float()  # (2, max_length, 66)

        return ret
    
    def __len__(self):
        return len(self.dances['pos3dl'])
    
    def __getitem__(self, index):
        dances_motion = self.dances
        
        # 获取名称, 运动和音乐数据
        name = self.names[index]
        lmotion = dances_motion['pos3dl'][index]  # 领导者运动，形状 (seq_len, 75)
        fmotion = dances_motion['pos3df'][index]  # 跟随者运动，形状 (seq_len, 75)
        music = self.dances['music'][index]    # 音乐特征，形状 (seq_len, music_dim)
        
        length = lmotion.shape[0]
        if not self.full_length:
            if length > self.max_length:
                idx = random.choice(list(range(0, length - self.max_length, 1)))
                gt_length = self.max_length
            else:
                # raise ValueError(f"length {length} is less than max_length {self.max_length}")
                idx = 0
                gt_length = min(length - idx, self.max_length)
            lmotion = lmotion[idx:idx + gt_length]
            fmotion = fmotion[idx:idx + gt_length]
            music = music[idx:idx + gt_length]
        
        gt_length = len(fmotion)
        
        # 零填充到 max_length
        if gt_length < self.max_length:
            padding_len = self.max_length - gt_length
            D = lmotion.shape[1]
            padding_zeros = np.zeros((padding_len, D))
            lmotion = np.concatenate((lmotion, padding_zeros), axis=0)
            fmotion = np.concatenate((fmotion, padding_zeros), axis=0)

            D_music = music.shape[1]
            padding_zeros_music = np.zeros((padding_len, D_music))
            music = np.concatenate((music, padding_zeros_music), axis=0)

        # Convert NumPy arrays to PyTorch tensors
        lmotion = torch.from_numpy(lmotion).float()
        fmotion = torch.from_numpy(fmotion).float()
        music = torch.from_numpy(music).float()
        gt_length = torch.tensor(gt_length).long()    
        
        if not (lmotion.shape[0] == fmotion.shape[0] == music.shape[0]):
            print(f"length {lmotion.shape[0]} is not equal to {fmotion.shape[0]} or {music.shape[0]}")
        assert lmotion.shape[0] == fmotion.shape[0] == music.shape[0]
        
        return name, music, lmotion, fmotion, gt_length