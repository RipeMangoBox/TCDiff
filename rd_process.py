import torch
from collections import OrderedDict
from dataset.dd100lf_all2 import DD100lfAll2
from tqdm import tqdm
import os
import numpy as np
from intergen_vis import generate_one_sample
from easydict import EasyDict
from scipy.ndimage import gaussian_filter, gaussian_filter1d

def batch_data_process(batch_data, normalizer = None, device = None):
    name, music, lmotion, fmotion, motion_lens = batch_data

    batch = OrderedDict({})
    batch["wavnames"] = name
    if normalizer is not None:
        batch["fmotion"] = normalizer.normalize(fmotion.to(device))
        batch["lmotion"] = normalizer.normalize(lmotion.to(device))
    else:
        batch["fmotion"] = fmotion.to(device)
        batch["lmotion"] = lmotion.to(device)
    batch["music"] = music.to(device)
    return batch

def motion_process_backward(lmotion, fmotion, duration=None):
    lmotion = lmotion.cpu().data.numpy()
    fmotion = fmotion.cpu().data.numpy()
    return lmotion[0], fmotion[0]

def DD100lf_data_loader(data_cfg, num_workers, shuffle=False, full_length=False):
    dataset = DD100lfAll2(
        data_cfg.music_root, 
        data_cfg.data_root, 
        split=data_cfg.split, 
        full_length=full_length
    )
    
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=data_cfg.batch_size if not full_length else 1,
        num_workers=num_workers,
        pin_memory=True,
        shuffle=shuffle,
        drop_last=True,
        # sampler=sampler,
        # collate_fn=custom_collate_fn,
    )
    
def DD100lf_dl(num_workers=16, full_length=False):
    data_cfg = EasyDict({
        'train': {
            'music_root': '../ReactDance/data_lazy/music',
            'data_root': '../ReactDance/data_lazy/motion',
            'motion_data_txt': None,
            'split': 'train',
            'batch_size': 16
        },
        'val': {
            'music_root': '../ReactDance/data_lazy/music',
            'data_root': '../ReactDance/data_lazy/motion',
            'motion_data_txt': None,
            'split': 'test',
            'batch_size': 16
        },
        'test': {
            'music_root': '../ReactDance/data_lazy/music',
            'data_root': '../ReactDance/data_lazy/motion',
            'motion_data_txt': None,
            'split': 'test',
            'batch_size': 1
        }
    })
    full_length = False
    
    train_dl = DD100lf_data_loader(data_cfg.train, num_workers, shuffle=True, full_length=full_length)    
    
    # segmented val data
    val_dl = DD100lf_data_loader(data_cfg.val, num_workers, shuffle=False)
    # whole test data
    whole_test_dl = DD100lf_data_loader(data_cfg.test, num_workers, shuffle=False, full_length=True)
    
    return train_dl, val_dl, whole_test_dl

def calc_duet_joints_vq(model, device, test_dl, mask):
    device = device
    with torch.no_grad():
        model = model.to(device).eval()
        print("GestureLSM_vq Duet Eval...", flush=True)
        followers = []
        leaders = []
        fnames = []
        for i, batch_data in enumerate(tqdm(test_dl, desc=f'[*] calc_duet_joints')):
            batch = batch_data_process(batch_data, normalizer=None)
            gt_motion = batch['fmotion'][..., mask]
            pred_motion, loss_commit, perplexity = model(gt_motion).values()
            
            fmotion = batch['fmotion']
            lmotion = batch['lmotion']
            fmotion[..., mask] = pred_motion
            lpos, fpos = motion_process_backward(lmotion, fmotion, duration=None)
            
            followers.append(fpos)
            leaders.append(lpos)
            fnames.append(batch_data[0][0])
        return leaders, followers, fnames

def calc_duet_joints_vq_full(upper_model, lower_model, device, test_dl, upper_mask, lower_mask):
    device = device
    with torch.no_grad():
        upper_model = upper_model.to(device).eval()
        lower_model = lower_model.to(device).eval()
        print("GestureLSM_vq Duet Eval...", flush=True)
        followers = []
        leaders = []
        fnames = []
        for i, batch_data in enumerate(tqdm(test_dl, desc=f'[*] calc_duet_joints')):
            batch = batch_data_process(batch_data, normalizer=None)
            gt_upper_motion = batch['fmotion'][..., upper_mask]
            gt_lower_motion = batch['fmotion'][..., lower_mask]
            pred_upper_motion, loss_commit, perplexity = upper_model(gt_upper_motion).values()
            pred_lower_motion, loss_commit, perplexity = lower_model(gt_lower_motion).values()
            
            fmotion = batch['fmotion']
            lmotion = batch['lmotion']
            fmotion[..., upper_mask] = pred_upper_motion
            fmotion[..., lower_mask] = pred_lower_motion
            lpos, fpos = motion_process_backward(lmotion, fmotion, duration=None)
            
            followers.append(fpos)
            leaders.append(lpos)
            fnames.append(batch_data[0][0])
        return leaders, followers, fnames
    
def eval_during_training(model, duet_metric_calc, device, test_dl, mask, writer, logger, global_step):
    jointsl_list, jointsf_list, fnames = calc_duet_joints_vq(model, device, test_dl, mask)
    duet_metrics = duet_metric_calc.eval_duet_metrics(jointsl_list, jointsf_list, fnames)
    
    for key, value in duet_metrics.items():
        writer.add_scalar(f'Eval/{key}', value, global_step)
        logger.info(f'Eval. Iter {global_step} : {key} {value:.2f}')
    return duet_metrics

def synthesis_and_vis_vq(epoch, expdir, model, device, val_dl, test_dl, mask, normalizer=None):
    def synthesis_and_vis_one_sample(model, epoch, dataset, dataset_name, device, sample_num, seq_len):
        # 对test_dl中第0下标的数据进行合成和可视化
        device = device
        with torch.no_grad():
            model = model.to(device).eval()
            for i, batch_data in enumerate(dataset):
                # batch_data: name, text, motion1, motion2, motion_lens
                name, text, motion1, motion2, motion_lens = batch_data
                fname = name[0] if isinstance(name, (list, tuple)) else str(name)
                
                batch = batch_data_process(batch_data, normalizer)
                gt_motion = batch['fmotion'][..., mask]
                pred_motion, loss_commit, perplexity = model(gt_motion).values()
                fmotion = batch['fmotion']
                lmotion = batch['lmotion']
                fmotion[..., mask] = pred_motion
                lpos, fpos = motion_process_backward(lmotion, fmotion, duration=seq_len)
                
                # 保存和可视化
                video_dir = os.path.join(expdir, "videos")
                os.makedirs(video_dir, exist_ok=True)
                
                # save video
                motion_both = [lpos[:seq_len], fpos[:seq_len]]
                lmask = [0] * 22
                fmask = [int(i in [v//3 for v in mask[::3]]) for i in range(22)]
                real_generated_motion_mask = [lmask, fmask]
                generate_one_sample(motion_both, real_generated_motion_mask, f"{fname}_{dataset_name}_epoch{epoch}", video_dir)
                
                if i >= sample_num:
                    break
    synthesis_and_vis_one_sample(model, epoch, val_dl, 'val', device, sample_num=2, seq_len=300, normalizer=normalizer)
    synthesis_and_vis_one_sample(model, epoch, test_dl, 'test', device, sample_num=1, seq_len=450, normalizer=normalizer)
        

def synthesis_and_vis_vq_full(expdir, upper_model, lower_model, device, val_dl, test_dl, upper_mask, lower_mask):
    def synthesis_and_vis_one_sample(upper_model, lower_model, dataset, dataset_name, device, sample_num, seq_len):
        # 对test_dl中第0下标的数据进行合成和可视化
        device = device
        with torch.no_grad():
            upper_model = upper_model.to(device).eval()
            lower_model = lower_model.to(device).eval()
            for i, batch_data in enumerate(dataset):
                # batch_data: name, text, motion1, motion2, motion_lens
                name, text, motion1, motion2, motion_lens = batch_data
                fname = name[0] if isinstance(name, (list, tuple)) else str(name)
                
                batch = batch_data_process(batch_data)
                gt_motion = batch['fmotion']
                pred_upper_motion, loss_commit, perplexity = upper_model(gt_motion[..., upper_mask]).values()
                pred_lower_motion, loss_commit, perplexity = lower_model(gt_motion[..., lower_mask]).values()
                fmotion = batch['fmotion']
                lmotion = batch['lmotion']
                fmotion[..., upper_mask] = pred_upper_motion
                fmotion[..., lower_mask] = pred_lower_motion
                lpos, fpos = motion_process_backward(lmotion, fmotion, duration=seq_len)
                
                # 保存和可视化
                video_dir = os.path.join(expdir, "videos")
                os.makedirs(video_dir, exist_ok=True)
                
                # save video
                motion_both = [lpos[:seq_len], fpos[:seq_len]]
                lmask = [0] * 22
                fmask = [int(i in [v//3 for v in upper_mask[::3]]) or int(i in [v//3 for v in lower_mask[::3]]) for i in range(22)]
                real_generated_motion_mask = [lmask, fmask]
                generate_one_sample(motion_both, real_generated_motion_mask, f"{fname}_{dataset_name}", video_dir)
                
                if i >= sample_num:
                    break
    synthesis_and_vis_one_sample(upper_model, lower_model, val_dl, 'val', device, sample_num=2, seq_len=128)
    synthesis_and_vis_one_sample(upper_model, lower_model, test_dl, 'test', device, sample_num=1, seq_len=600)
        
def synthesis(result_dir, model, device, test_dl, mask, normalizer=None):
    def synthesis_and_vis(seqlen):
        # 对test_dl中第0下标的数据进行合成和可视化
        device = device
        # 保存和可视化
        expdir = result_dir
        synthdir = os.path.join(expdir, "test_synthesis_log")
        npy_dir = os.path.join(synthdir, f"npy/{seqlen}")
        video_dir = os.path.join(synthdir, f"videos/{seqlen}")
        os.makedirs(npy_dir, exist_ok=True)
        os.makedirs(video_dir, exist_ok=True)
        
        with torch.no_grad():
            model = model.to(device).eval()
            for batch_data in test_dl:
                # batch_data: name, text, motion1, motion2, motion_lens
                name, text, motion1, motion2, motion_lens = batch_data
                fname = name[0] if isinstance(name, (list, tuple)) else str(name)
                
                batch = batch_data_process(batch_data, normalizer)
                output = model.forward_test(batch)['output']
                B, T, D = output.shape
                lmotion_output, fmotion_output = torch.split(output, [D//2, D//2], dim=-1)
                lpos, fpos = motion_process_backward(lmotion_output, fmotion_output, duration=seqlen)
                
                # save npy
                np.save(os.path.join(npy_dir, f"{fname}.npy"), np.concatenate([lpos, fpos], axis=1))
                # save video
                motion_both = [lpos, fpos]
                generate_one_sample(motion_both, f"{fname}", video_dir)
                
    synthesis_and_vis(seqlen=128, normalizer=normalizer)
    synthesis_and_vis(seqlen=None, normalizer=normalizer)
        
def save_pos3d(posf, posl, evaldir, fname, suffix='pos3d_npy'):
    save_folder = os.path.join(evaldir, suffix)
    os.makedirs(save_folder, exist_ok=True)
    
    np.save(os.path.join(save_folder, fname + '_00'), posf)
    np.save(os.path.join(save_folder, fname + '_01'), posl)
    
    
def motion_temporal_filter(motion, filter_type, filter_kwargs):
    """
    对 motion 数据进行时间维度上的平滑处理。
    
    参数:
        motion (np.ndarray): 输入的 motion 数据，形状为 (b, n, d)，其中
                             b 是 batch_size，n 是时序长度，d 是关节位置维度。
        filter_type (str): 平滑滤波器类型，支持 "gaussian" 和 "savgol_filter"。
        sigma (float): 高斯滤波的标准差（仅在 filter_type="gaussian" 时使用）。
    
    返回:
        np.ndarray: 平滑后的 motion 数据，形状与输入相同。
    """
    b, n, d = motion.shape  # 获取 batch_size、时序长度和维度
    
    ret_tensor = torch.is_tensor(motion)
    if ret_tensor:
        device = motion.device
        motion = motion.cpu().numpy()
    
    if filter_type == "gaussian":
        # 使用高斯滤波对时间维度进行平滑
        smoothed_motion = gaussian_filter1d(motion, sigma=filter_kwargs["sigma"], axis=1, mode='nearest')
    
    elif filter_type == "savgol":
        # 使用 Savitzky-Golay 滤波对时间维度进行平滑
        # 注意：savgol_filter 不支持直接对多维数组操作，因此需要逐 batch 处理
        window_length = filter_kwargs["window_length"]  # 窗口长度
        polyorder = filter_kwargs["polyorder"]      # 多项式阶数
        
        # 初始化输出数组
        smoothed_motion = np.zeros_like(motion)
        
        for i in range(b):  # 对每个 batch 分别处理
            smoothed_motion[i] = savgol_filter(motion[i], window_length=window_length, polyorder=polyorder, axis=0)
    
    else:
        raise ValueError(f"Unsupported filter type: {filter_type}")
    
    if ret_tensor:
        smoothed_motion = torch.from_numpy(smoothed_motion).to(device)
    return smoothed_motion