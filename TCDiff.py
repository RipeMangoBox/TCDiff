import multiprocessing
import os
import pickle
from functools import partial
from pathlib import Path

import torch
import torch.nn.functional as F
# import wandb
from torch.utils.tensorboard import SummaryWriter
from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.state import AcceleratorState
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset.group_dataset import AIOZDataset
from dataset.preprocess import increment_path
from model.adan import Adan
from model.diffusion import GaussianDiffusion
from model.model import DanceDecoder
from vis import SMPLSkeleton
from dataset.dd100lf_all2 import DD100lfAll2

## train model
from TrajDecoder.model.traj_model import *
from TrajDecoder.dataset.traj_dataset import *
from TrajDecoder.vis import render_sample as render_traj_sample
import TrajDecoder.options.option_traj as option_traj 
from TrajDecoder.utils.utils_model import kalman_smooth_batch
from rd_process import batch_data_process
import time
    
# To resolve CUDA errors, execute unset LD_LIBRARY_PATH. See this blog post for more information. https://blog.csdn.net/BetrayFree/article/details/133868929

def wrap(x):
    return {f"module.{key}": value for key, value in x.items()}


def maybe_wrap(x, num):
    return x if num == 1 else wrap(x)


class TCDiff:
    def __init__(
        self,
        checkpoint_path="",
        EMA=True,
        learning_rate=4e-4,
        weight_decay=0.02,
        required_dancer_num = 3, 
        window_size = 150,
        opt = None,
    ):
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=False)
        self.accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])
        state = AcceleratorState()
        num_processes = state.num_processes

        pos_dim = 3 * 22
        # addition_dim = 0  # Reminder: beta and meta parameters are currently excluded from the representation
        repr_dim = pos_dim + 4  # (+4) accounts for additional features such as global controls;

        self.repr_dim = repr_dim
        self.required_dancer_num = required_dancer_num
        feature_dim = 4800 # music feature dim
        self.horizon = horizon = window_size
        self.accelerator.wait_for_everyone()

        # load datasets
        train_tensor_dataset_path = os.path.join(
            opt.processed_data_dir, f"train_tensor_dataset.pkl"
        )
        test_tensor_dataset_path = os.path.join(
            opt.processed_data_dir, f"test_tensor_dataset.pkl"
        )
        if (
            not opt.no_cache
            and os.path.isfile(train_tensor_dataset_path)
            and os.path.isfile(test_tensor_dataset_path)
        ):
            self.train_dataset = pickle.load(open(train_tensor_dataset_path, "rb"))
            self.test_dataset = pickle.load(open(test_tensor_dataset_path, "rb"))
        else:
            self.train_dataset = DD100lfAll2(
                split='train', full_length=False
            )
            self.test_dataset = DD100lfAll2(
                split='test', full_length=False, normalizer=self.train_dataset.normalizer
            )

        self.normalizer = self.train_dataset.normalizer
        
        checkpoint = None
        if checkpoint_path != "":
            checkpoint = torch.load(
                checkpoint_path, map_location=self.accelerator.device
            )
            self.normalizer = checkpoint["normalizer"]

        model = DanceDecoder( 
             nfeats=repr_dim,
             seq_len=horizon,
             latent_dim=512,
             ff_size=1024,
             num_layers=8, 
             num_heads=8,
             dropout=0.1,
             cond_feature_dim=feature_dim,
             activation=F.gelu,
             required_dancer_num = required_dancer_num,
         )

        diffusion = GaussianDiffusion(
            model,
            horizon,
            repr_dim,
            schedule="cosine",
            n_timestep=1000,
            predict_epsilon=False,
            loss_type="l2",
            use_p2=False,
            cond_drop_prob=0.25,
            guidance_weight=2,
            normalizer=self.normalizer,
        )

        print(
            "Model has {} parameters".format(sum(y.numel() for y in model.parameters()))
        )

        self.model = self.accelerator.prepare(model)
        self.diffusion = diffusion.to(self.accelerator.device)
        optim = Adan(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.optim = self.accelerator.prepare(optim)

        if checkpoint_path != "":
            self.model.load_state_dict(
                maybe_wrap(
                    checkpoint["ema_state_dict" if EMA else "model_state_dict"],
                    num_processes,
                ),
                strict=False
            )
            print(f"loading ckpt from {checkpoint_path}")

    def eval(self):
        self.diffusion.eval()

    def train(self):
        self.diffusion.train()

    def prepare(self, objects):
        return self.accelerator.prepare(*objects)

    def train_loop(self, opt): 
        num_cpus = multiprocessing.cpu_count()
        train_data_loader = DataLoader(
            self.train_dataset,
            batch_size=opt.batch_size,
            shuffle=True,
            num_workers=8,
            pin_memory=True,
            drop_last=True,
        )
        test_data_loader = DataLoader(
            self.test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=8,
            pin_memory=True,
            drop_last=True,
        )
        
        self.train_data_loader = self.accelerator.prepare(train_data_loader)
        self.test_data_loader = self.accelerator.prepare(test_data_loader)
        self.normalizer = self.train_data_loader.dataset.normalizer
        
        # --- No changes needed here ---
        if self.accelerator.is_main_process:
            save_dir = str(increment_path(Path(opt.project) / opt.exp_name))
            opt.exp_name = save_dir.split("/")[-1]
            save_dir = Path(save_dir)
            self.save_dir = save_dir
            wdir = save_dir / "weights"
            wdir.mkdir(parents=True, exist_ok=True)
            self.writer = SummaryWriter(os.path.join(wdir, "tensorboard"))

        self.accelerator.wait_for_everyone()
        print("Begin Training")
        for epoch in range(1, opt.epochs + 1):
            avg_loss = 0
            avg_vloss = 0
            avg_footloss = 0
            
            self.train()

            # Create the tqdm pbar only on the main process
            if self.accelerator.is_main_process:
                pbar = tqdm(train_data_loader, position=0, desc=f"Epoch {epoch}/{opt.epochs}")
            else:
                pbar = train_data_loader

            for step, batch in enumerate(pbar):
                # Calculate a global step for a continuous x-axis in TensorBoard
                global_step = (epoch - 1) * len(train_data_loader) + step

                batch_data = batch_data_process(batch, self.normalizer, device=self.accelerator.device)
                x, lmotion, music, wavnames = batch_data["fmotion"], batch_data["lmotion"], batch_data["music"], batch_data["wavnames"]
                x = x.unsqueeze(1)
                total_loss, (loss, v_loss, foot_loss) = self.diffusion( 
                    x, lmotion, music, t_override=None 
                )
                
                # --- CHANGE 1: Log each loss individually with add_scalar ---
                if self.accelerator.is_main_process:
                    # Update tqdm progress bar
                    loss_dict = {
                        "loss": loss.item(),
                        "v_loss": v_loss.item(),
                        "foot_loss": foot_loss.item(),
                    }
                    pbar.set_postfix(loss_dict)
                    
                    # Log scalars to TensorBoard with a hierarchical tag
                    self.writer.add_scalar('train/loss', loss.item(), global_step)
                    self.writer.add_scalar('train/v_loss', v_loss.item(), global_step)
                    self.writer.add_scalar('train/foot_loss', foot_loss.item(), global_step)

                self.optim.zero_grad()
                self.accelerator.backward(total_loss)
                self.optim.step()

                if self.accelerator.is_main_process:
                    avg_loss += loss.item() # Use .item() to avoid holding onto the graph
                    avg_vloss += v_loss.item()
                    avg_footloss += foot_loss.item()
                    if step % opt.ema_interval == 0:
                        self.diffusion.ema.update_model_average(
                            self.diffusion.master_model, self.diffusion.model
                        )
                        
            if (epoch % opt.save_interval) == 0:
                self.accelerator.wait_for_everyone()
                if self.accelerator.is_main_process:
                    self.eval()
                    
                    # --- CHANGE 2: Log epoch-level average losses ---
                    avg_loss /= len(train_data_loader)
                    avg_vloss /= len(train_data_loader)
                    avg_footloss /= len(train_data_loader)
                    
                    # Log each average loss for the epoch
                    self.writer.add_scalar('epoch_avg/train_loss', avg_loss, epoch)
                    self.writer.add_scalar('epoch_avg/train_vloss', avg_vloss, epoch)
                    self.writer.add_scalar('epoch_avg/train_footloss', avg_footloss, epoch)

                    print(f"Epoch {epoch} Average Losses: Train={avg_loss:.4f}, V={avg_vloss:.4f}, Foot={avg_footloss:.4f}")
                    
                    # --- The rest of your saving and rendering logic ---
                    ckpt = {
                        "ema_state_dict": self.diffusion.master_model.state_dict(),
                        "model_state_dict": self.accelerator.unwrap_model(self.model).state_dict(),
                        "optimizer_state_dict": self.optim.state_dict(),
                        "normalizer": self.normalizer,
                    }
                    torch.save(ckpt, os.path.join(wdir, f"train-{epoch}.pt"))
                    
                    # generate a sample
                    render_count = 1
                    shape = (render_count, self.horizon, self.repr_dim)
                    print("Generating Sample")
                    # draw a cond from the test dataset
                    batch_data = batch_data_process(next(iter(self.test_data_loader)), self.normalizer, device=self.accelerator.device)
                    fmotion, lmotion, music, wavnames = batch_data["fmotion"], batch_data["lmotion"], batch_data["music"], batch_data["wavnames"]
                    x = fmotion.unsqueeze(1)
                    
                    # x_traj_xy = x[:,0,:,[4,4+1]] # [*, 150, 2]
                    # l_traj_xy = lmotion[:,:,[4,4+1]] # [*, 150, 2]
                    # bs, seq, c = x_traj_xy.shape
                    # x_traj = torch.zeros(bs, 2, seq, 3).to(x_traj_xy) # Note: Due to some historical baggage, we kept the option to input full xyz coordinates...
                    # x_traj[:,0,:,[0,1]] = x_traj_xy[:,:,[0,1]] 
                    # x_traj[:,1,:,[0,1]] = l_traj_xy[:,:,[0,1]] 
                    
                    x_traj_xy = x[:,0,:,[4,4+1]] # [*, 150, 2]
                    # l_traj_xy = lmotion[:,:,[4,4+1]] # [*, 150, 2]
                    bs, seq, c = x_traj_xy.shape
                    x_traj = torch.zeros(bs, 1, seq, 3).to(x_traj_xy) # Note: Due to some historical baggage, we kept the option to input full xyz coordinates...
                    x_traj[:,0,:,[0,1]] = x_traj_xy[:,:,[0,1]] 
                    # x_traj[:,1,:,[0,1]] = l_traj_xy[:,:,[0,1]] 

                    self.diffusion.render_sample( 
                        shape,
                        lmotion,
                        music,
                        self.normalizer,
                        epoch,
                        os.path.join(self.save_dir, "renders"),
                        name=wavnames,
                        sound=True,
                        required_dancer_num = self.required_dancer_num,
                        x_0 = None, 
                    )
                    print(f"[MODEL SAVED at Epoch {epoch}]")
        
        if self.accelerator.is_main_process:
            self.writer.close()


    # def given_trajectory_generation_loop(self, opt): 
    #     self.train_data_loader = self.accelerator.prepare(self.train_data_loader)
    #     self.test_data_loader = self.accelerator.prepare(self.test_data_loader)
    #     self.normalizer = self.train_data_loader.dataset.normalizer
        
    #     # boot up multi-gpu training. test dataloader is only on main process
    #     load_loop = (
    #         partial(tqdm, position=1, desc="Batch"),
    #         lambda x: x
    #     )

    #     render_count = 30 
    #     shape = (render_count, self.horizon*self.required_dancer_num, self.repr_dim)
        
    #     print("Begin validation with given trajectories")
    #     self.eval()
    #     for epoch in range(1, opt.epochs + 1):
    #         # draw a cond from the test dataset
    #         batch_data = batch_data_process(next(iter(self.train_data_loader)), self.normalizer, device=self.accelerator.device)
    #         x, lmotion, music, wavnames = batch_data["fmotion"], batch_data["lmotion"], batch_data["music"], batch_data["wavnames"]
    #         print("Generating Sample")
    #         x_traj_xy = x[:,:,[4,4+1]] # [*, 150, 2]
    #         l_traj_xy = lmotion[:,:,[4,4+1]] # [*, 150, 2]
    #         bs, seq, c = x_traj_xy.shape
    #         x_traj = torch.zeros(bs, 2, seq, 3).to(x_traj_xy) # Note: Due to some historical baggage, we kept the option to input full xyz coordinates...
    #         x_traj[:,0,:,[0,1]] = x_traj_xy[:,:,[0,1]] 
    #         x_traj[:,1,:,[0,1]] = l_traj_xy[:,:,[0,1]] 

    #         self.diffusion.render_sample( 
    #             shape,
    #             music[:render_count],
    #             lmotion[:render_count],
    #             self.normalizer,
    #             epoch,
    #             os.path.join(opt.render_dir, "Given_Train_" + opt.exp_name),
    #             name=wavnames[:render_count],
    #             sound=True,
    #             required_dancer_num= self.required_dancer_num,
    #             x_0 = x_traj[:render_count].permute(0,2,1,3).reshape(render_count,shape[1], 3), 
    #         )
    #         print(f"[TRAIN-RENDER SAVED at Epoch {epoch}]")

            
    #         shape = (render_count, self.horizon*self.required_dancer_num, self.repr_dim)
    #         print("Generating Sample")
    #         # draw a cond from the test dataset
    #         batch_data = batch_data_process(next(iter(self.test_data_loader)), self.normalizer, device=self.accelerator.device)
    #         x, lmotion, music, wavnames = batch_data["fmotion"], batch_data["lmotion"], batch_data["music"], batch_data["wavnames"]

    #         x_traj_xy = x[:,:,[4,4+1]] # [*, 150, 2]
    #         l_traj_xy = lmotion[:,:,[4,4+1]] # [*, 150, 2]
    #         bs, seq, c = x_traj_xy.shape
    #         x_traj = torch.zeros(bs, 2, seq, 3).to(x_traj_xy) # Note: Due to some historical baggage, we kept the option to input full xyz coordinates...
    #         x_traj[:,0,:,[0,1]] = x_traj_xy[:,:,[0,1]] 
    #         x_traj[:,1,:,[0,1]] = l_traj_xy[:,:,[0,1]] 

    #         self.diffusion.render_sample( 
    #             shape,
    #             music[:render_count],
    #             lmotion[:render_count],
    #             self.normalizer,
    #             epoch,
    #             os.path.join(opt.render_dir, "Given_Test_" + opt.exp_name),
    #             name=wavnames[:render_count],
    #             sound=True,
    #             required_dancer_num= self.required_dancer_num,
    #             x_0 = x_traj[:render_count].permute(0,2,1,3).reshape(render_count,shape[1], 3), # [2, seq, 3, 2] 
    #         )
    #         print(f"[VAL-RENDER SAVED at Epoch {epoch}]")


    # def test_loop(self, opt): 
    #     train_tensor_dataset_path = os.path.join(
    #         opt.processed_data_dir, f"train_tensor_dataset.pkl"
    #     )
    #     test_tensor_dataset_path = os.path.join(
    #         opt.processed_data_dir, f"test_tensor_dataset.pkl"
    #     )
    #     if (
    #         not opt.no_cache
    #         and os.path.isfile(train_tensor_dataset_path) 
    #         and os.path.isfile(test_tensor_dataset_path)
    #     ):
    #         train_dataset = pickle.load(open(train_tensor_dataset_path, "rb"))
    #         test_dataset = pickle.load(open(test_tensor_dataset_path, "rb"))
    #     else:
    #         train_dataset = AIOZDataset(
    #             data_path=opt.data_path,
    #             backup_path=opt.processed_data_dir,
    #             train=True,
    #             force_reload=opt.force_reload,
    #             required_dancer_num = self.required_dancer_num, 
    #         )
    #         test_dataset = AIOZDataset(
    #             data_path=opt.data_path,
    #             backup_path=opt.processed_data_dir,
    #             train=False,
    #             normalizer=train_dataset.normalizer,
    #             force_reload=opt.force_reload,
    #             required_dancer_num = self.required_dancer_num, 
    #         )
    #         # cache the dataset in case
    #         pickle.dump(train_dataset, open(train_tensor_dataset_path, "wb"))
    #         pickle.dump(test_dataset, open(test_tensor_dataset_path, "wb"))

    #     # set normalizer
    #     self.normalizer = test_dataset.normalizer

    #     # data loaders
    #     # decide number of workers based on cpu count
    #     num_cpus = multiprocessing.cpu_count()
    #     train_data_loader = DataLoader(
    #         train_dataset,
    #         batch_size=opt.batch_size,
    #         shuffle=True,
    #         num_workers=min(int(num_cpus * 0.75), 32),
    #         pin_memory=True,
    #         drop_last=True,
    #     )
    #     test_data_loader = DataLoader(
    #         test_dataset,
    #         batch_size=opt.batch_size,
    #         shuffle=True,
    #         num_workers=2,
    #         pin_memory=True,
    #         drop_last=True,
    #     )

    #     # boot up multi-gpu training. test dataloader is only on main process
    #     load_loop = (
    #         partial(tqdm, position=1, desc="Batch"),
    #         lambda x: x
    #     )

    #     render_count = 30 
    #     shape = (render_count, self.horizon*self.required_dancer_num, self.repr_dim)

    #     ## init Trajectory Model
    #     trjm_args = option_traj.get_args_parser()
    #     torch.manual_seed(trjm_args.seed)
    #     window_size = trjm_args.window_size # align with training
    #     step = trjm_args.step
    #     traj_model = TrajDecoder(nfeats = trjm_args.nfeats, 
    #               trans_layer = trjm_args.trans_layer, 
    #               window_size = trjm_args.window_size,
    #               ) 
    #     if trjm_args.checkpoint is not None:
    #         ckpt = torch.load(opt.traj_checkpoint, map_location='cpu')
    #         traj_model.load_state_dict(ckpt['net'], strict=True) 
    #         print('loading checkpoint from {}'.format(opt.traj_checkpoint))
    #     traj_model.cuda().eval()
        
    #     print("Begin testing with generated trajectories")
    #     self.eval()
    #     for epoch in range(1, opt.epochs + 1):
    #         # draw a cond from the test dataset
    #         batch_data = batch_data_process(next(iter(self.train_data_loader)), self.normalizer, device=self.accelerator.device)
    #         x, lmotion, music, wavnames = batch_data["fmotion"], batch_data["lmotion"], batch_data["music"], batch_data["wavnames"]
    #         x = x.unsqueeze(1)
    #         print("Generating Sample")

    #         # Autoregressively generate the full trajectory sequence
    #         pre_list = []

    #         # Extract initial xy trajectory from input data
    #         x_traj_xy = x[:,:,[4,4+1]] # [*, 150, 2]
    #         l_traj_xy = lmotion[:,:,[4,4+1]] # [*, 150, 2]
    #         bs, seq, c = x_traj_xy.shape
    #         x_traj = torch.zeros(bs, 2, seq, 3).to(x_traj_xy) # Note: Due to some historical baggage, we kept the option to input full xyz coordinates...
    #         x_traj[:,0,:,[0,1]] = x_traj_xy[:,:,[0,1]] 
    #         x_traj[:,1,:,[0,1]] = l_traj_xy[:,:,[0,1]] 

    #         # Initialize the first window for trajectory prediction
    #         cond_traj = x_traj[:, :,:window_size,[0,1]] 
    #         pre_list.append(cond_traj) 
    #         cond_len = music.shape[1]

    #         # Slide a window over the cond features
    #         # cond sequence length is (window_size + step) * 2 because cond FPS is twice the motion FPS
    #         # Hence, move the cond window by step*2 each time
    #         for start in range(0, cond_len + 1-(window_size+step)*2, step*2):  
    #             # Predict the next trajectory segment
    #             pre_traj = traj_model(cond_traj, music[ :, start:start + (window_size+step) * 2], lmotion[ :, start:start + (window_size+step) * 2]) 
    #             cond_traj = pre_traj
    #             pre_list.append(pre_traj[:,:,-step:])
            
    #         # Concatenate all trajectory segments into a single sequence
    #         x_traj = torch.cat(pre_list,dim = 2) 

    #         # Optional: process trajectory with smoothing or constraints
    #         x_traj = kalman_smooth_batch(x_traj.cpu().detach().numpy())
    #         x_traj = torch.from_numpy(x_traj).to(dtype=x.dtype, device=x.device)
            
    #         # Pad the trajectory to 3D space by adding a zero z-coordinate
    #         bs, dn, seq, c = x_traj.shape
    #         x_traj_padding = torch.zeros(bs, dn, seq, 3).to(x_traj)
    #         x_traj_padding[:,:,:,[0,1]] = x_traj[:,:,:,[0,1]] 

    #         self.diffusion.render_sample( 
    #             shape,
    #             music[:render_count],
    #             lmotion[:render_count],
    #             self.normalizer,
    #             epoch,
    #             os.path.join(opt.render_dir, "TRAIN_" + opt.exp_name),
    #             name=wavnames[:render_count],
    #             sound=True,
    #             required_dancer_num= self.required_dancer_num,
    #             x_0 = x_traj_padding[:render_count].permute(0,2,1,3).reshape(render_count,shape[1], 3), 
    #         )
    #         print(f"[TRAIN-RENDER SAVED at Epoch {epoch}]")

            
    #         shape = (render_count, self.horizon*self.required_dancer_num, self.repr_dim)
    #         print("Generating Sample")
    #         # draw a cond from the test dataset
    #         batch_data = batch_data_process(next(iter(self.test_data_loader)), self.normalizer, device=self.accelerator.device)
    #         x, lmotion, music, wavnames = batch_data["fmotion"], batch_data["lmotion"], batch_data["music"], batch_data["wavnames"]
    #         x = x.unsqueeze(1)

    #         # Autoregressively generate the full trajectory sequence
    #         pre_list = []

    #         # Extract initial xy trajectory from input data
    #         x_traj_xy = x[:,:,[4,4+1]] # [*, 150, 2]
    #         l_traj_xy = lmotion[:,:,[4,4+1]] # [*, 150, 2]
    #         bs, seq, c = x_traj_xy.shape
    #         x_traj = torch.zeros(bs, 2, seq, 3).to(x_traj_xy) # Note: Due to some historical baggage, we kept the option to input full xyz coordinates...
    #         x_traj[:,0,:,[0,1]] = x_traj_xy[:,:,[0,1]] 
    #         x_traj[:,1,:,[0,1]] = l_traj_xy[:,:,[0,1]] 

    #         # Initialize the first window for trajectory prediction
    #         cond_traj = x_traj[:, :,:window_size,[0,1]] 
    #         pre_list.append(cond_traj) 
    #         cond_len = music.shape[1]

    #         # Slide a window over the cond features
    #         # cond sequence length is (window_size + step) * 2 because cond FPS is twice the motion FPS
    #         # Hence, move the cond window by step*2 each time
    #         for start in range(0, cond_len + 1-(window_size+step)*2, step*2):  
    #             # Predict the next trajectory segment
    #             pre_traj = traj_model(cond_traj, music[ :, start:start + (window_size+step) * 2], lmotion[ :, start:start + (window_size+step) * 2]) 
    #             cond_traj = pre_traj
    #             pre_list.append(pre_traj[:,:,-step:])
            
    #         # Concatenate all trajectory segments into a single sequence
    #         x_traj = torch.cat(pre_list,dim = 2) 

    #         # Optional: process trajectory with smoothing or constraints
    #         x_traj = kalman_smooth_batch(x_traj.cpu().detach().numpy())
    #         x_traj = torch.from_numpy(x_traj).to(dtype=x.dtype, device=x.device)

    #         # Pad the trajectory to 3D space by adding a zero z-coordinate
    #         bs, dn, seq, c = x_traj.shape
    #         x_traj_padding = torch.zeros(bs, dn, seq, 3).to(x_traj)
    #         x_traj_padding[:,:,:,[0,1]] = x_traj[:,:,:,[0,1]] 

    #         self.diffusion.render_sample( 
    #             shape,
    #             music[:render_count],
    #             lmotion[:render_count],
    #             self.normalizer,
    #             epoch,
    #             os.path.join(opt.render_dir, "TEST_" + opt.exp_name),
    #             name=wavnames[:render_count],
    #             sound=True,
    #             required_dancer_num= self.required_dancer_num,
    #             x_0 = x_traj_padding[:render_count].permute(0,2,1,3).reshape(render_count,shape[1], 3), # [2, seq, 3, 2] 
    #         )
    #         print(f"[TEST-RENDER SAVED at Epoch {epoch}]")



    # def render_sample( # Renders long motion sequences for testing or visualization.
    #     self, data_tuple, label, render_dir, render_count=-1, render=True, x_0 = None, render_len = 512
    # ):
    #     _, music, lmotion, wavname = data_tuple
    #     assert len(music.shape) == 3
    #     # Automatically determine the number of audio segments to render
    #     if render_count < 0: 
    #         render_count = len(music)
    #     # Define the shape of the output motion sequence:
    #     #   - batch size: render_count
    #     #   - sequence length: horizon * number of dancers
    #     #   - feature dimension: representation dimension per frame
    #     shape = (render_count, self.horizon*self.required_dancer_num, self.repr_dim)
    #     self.diffusion.render_sample(
    #         shape,
    #         music[:render_count],
    #         lmotion[:render_count],
    #         self.normalizer,
    #         label, # During training: current epoch; during test: 'test'
    #         render_dir,
    #         name=wavname[:render_count],
    #         sound=True,
    #         mode="long",
    #         render=render,
    #         x_0 = x_0,
    #         required_dancer_num = self.required_dancer_num,
    #         render_len = render_len,
    #     )

    def render_sample( # Renders long motion sequences for testing or visualization.
        self, data_tuple, label, render_dir, render_count=-1, render=True, x_0 = None, render_len = 512, full_lmotion=None, idx_list=None
    ):
        _, music, lmotion, wavname = data_tuple
        assert len(music.shape) == 3
        # Automatically determine the number of audio segments to render
        if render_count < 0: 
            render_count = len(music)
        # Define the shape of the output motion sequence:
        shape = (render_count, self.horizon*self.required_dancer_num, self.repr_dim)
        
        # Pass the new arguments to the diffusion model's renderer
        self.diffusion.render_sample(
            shape,
            lmotion[:render_count],
            music[:render_count],
            self.normalizer,
            label, # During training: current epoch; during test: 'test'
            render_dir,
            name=wavname[:render_count],
            sound=True,
            mode="long",
            render=render,
            x_0 = x_0,
            required_dancer_num = self.required_dancer_num,
            render_len = render_len,
            full_lmotion=full_lmotion,
            idx_list=idx_list,
        )
        
    def long_sample(self, epoch, save_dir, maxlen=None):
        """
        Generates and renders a long motion sequence by slicing a long audio clip,
        generating motion for each slice, and stitching the results.
        """
        seqlen_name = f"sample_len{maxlen}" if maxlen is not None else "full"
        Output_DIR = os.path.join(save_dir, "long", f"e{epoch}", seqlen_name)
        os.makedirs(Output_DIR, exist_ok=True)
        time_write_file = os.path.join(Output_DIR, f"time.txt")
        
        # 1. Load the test dataset for long-sequence generation
        test_dataset = DD100lfAll2(
                split='test', full_length=True, normalizer=self.train_dataset.normalizer
            )
        test_data_loader = DataLoader(
            test_dataset,
            batch_size=1,  # Process one long sample at a time
            shuffle=False,
            num_workers=8,
            pin_memory=True,
            drop_last=False,
        )
        test_data_loader = self.accelerator.prepare(test_data_loader)

        print("Generating Long Sample...")
        # 初始化计时器
        total_forward_time = 0.0
        
        # 2. Iterate through the test set to get a long audio/motion pair
        for step, batch in tqdm(enumerate(test_data_loader), desc=f'[*] synthesis'):
            batch_data = batch_data_process(batch, self.normalizer, device=self.accelerator.device)
            _, lmotion, music, wavnames = batch_data["fmotion"], batch_data["lmotion"], batch_data["music"], batch_data["wavnames"]
            
            if maxlen is None:
                sample_len = lmotion.shape[1]  # 使用完整长度
            else:
                sample_len = min(maxlen, lmotion.shape[1])

            # raise Exception("stop")
            # lmotion = lmotion[:, :sample_len]
            # music = music[:, :sample_len]
            
            # ⏱️ 开始计时
            start_time = time.perf_counter()
            
            # Initialize empty lists to hold the overlapping slices and their indices
            lmotion_slices = []
            music_slices = []
            idx_list = [] # <-- 1. Initialize the index list

            # The stride for a 50% overlap, required by the stitching logic
            block_length = self.train_dataset.max_length
            stride = block_length // 2

            # Create overlapping slices
            for i in range(0, music.shape[1] - block_length + 1, stride):
                music_slices.append(music[0, i:i + block_length])
                lmotion_slices.append(lmotion[0, i:i + block_length])
                idx_list.append(i)

            # Add the final segment to ensure the entire sequence is covered
            if music.shape[1] % stride != 0:
                start_idx = music.shape[1] - block_length
                music_slices.append(music[0, -block_length:])
                lmotion_slices.append(lmotion[0, -block_length:])
                idx_list.append(start_idx)

            # Stack the list of slices into a single batch tensor
            # Shape: (num_slices, block_length, feature_dim)
            lmotion_batch = torch.stack(lmotion_slices, dim=0)
            music_batch = torch.stack(music_slices, dim=0)

            # The number of slices is our new batch size for rendering
            render_count = lmotion_batch.shape[0]

            # --- Call the Rendering Function ---
            current_wavname = [wavnames[0]]
            data_tuple = (None, music_batch, lmotion_batch, current_wavname)
            
            # print(f"Rendering {current_wavname[0]} from {render_count} overlapping slices.")
            
            self.render_sample(
                data_tuple,
                epoch, # Use epoch as the label for file naming
                Output_DIR,
                render_count=render_count,
                render=True, # Ensure rendering happens
                full_lmotion=lmotion, # Pass the original for comparison if needed
                idx_list=idx_list,
                render_len=sample_len,
            )
            
            # ⏱️ 结束计时
            end_time = time.perf_counter()
            # 累加耗时
            total_forward_time += (end_time - start_time)
            
        # 📊 打印总耗时
        with open(time_write_file, "w") as f:
            f.write(f"Total time: {total_forward_time:.4f} seconds, average forward time: {total_forward_time / len(test_dataset)} seconds")
        print(f"[INFO] Total time: {total_forward_time:.4f} seconds, average forward time: {total_forward_time / len(test_dataset)} seconds")
            