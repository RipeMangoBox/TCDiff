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

## train model
from TrajDecoder.model.traj_model import *
from TrajDecoder.dataset.traj_dataset import *
from TrajDecoder.vis import render_sample as render_traj_sample
import TrajDecoder.options.option_traj as option_traj 
from TrajDecoder.utils.utils_model import kalman_smooth_batch
from rd_process import batch_data_process, DD100lf_dl
    
# To resolve CUDA errors, execute unset LD_LIBRARY_PATH. See this blog post for more information. https://blog.csdn.net/BetrayFree/article/details/133868929

def wrap(x):
    return {f"module.{key}": value for key, value in x.items()}


def maybe_wrap(x, num):
    return x if num == 1 else wrap(x)


class TCDiff:
    def __init__(
        self,
        checkpoint_path="",
        normalizer=None,
        EMA=True,
        learning_rate=4e-4,
        weight_decay=0.02,
        required_dancer_num = 3, 
        window_size = 150,
        split_file = None,
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
        self.split_file = split_file
        feature_dim = 54 # music feature dim
        self.horizon = horizon = window_size
        self.accelerator.wait_for_everyone()

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

    def train_loop(self, opt, train_dataloader, test_dataloader): 
        train_data_loader = self.accelerator.prepare(train_dataloader)
        test_data_loader = self.accelerator.prepare(test_dataloader)
        self.normalizer = train_dataloader.dataset.normalizer
        
        self.writer = SummaryWriter(log_dir=opt.save_dir)
        
        # boot up multi-gpu training. test dataloader is only on main process
        # load_loop = (
        #     partial(tqdm, position=0)
        #     if self.accelerator.is_main_process
        #     else lambda x, **kwargs: x
        # )
        
        # 包装 dataloader 得到 tqdm 迭代器（仅主进程）
        if self.accelerator.is_main_process:
            pbar = tqdm(train_data_loader, position=1, desc="Batch")
        else:
            pbar = train_data_loader  # 非主进程不包装

        if self.accelerator.is_main_process:
            save_dir = str(increment_path(Path(opt.project) / opt.exp_name))
            opt.exp_name = save_dir.split("/")[-1]
            # wandb.init(project=opt.wandb_pj_name, name=opt.exp_name)
            save_dir = Path(save_dir)
            wdir = save_dir / "weights" # save ckpt path
            wdir.mkdir(parents=True, exist_ok=True)
            self.writer = SummaryWriter(os.path.join(wdir, "tensorboard"))

        self.accelerator.wait_for_everyone()
        print("Begin Traning")
        for epoch in range(1, opt.epochs + 1):
            avg_loss = 0
            avg_vloss = 0
            avg_footloss = 0
            
            self.train()
            # for step, batch in enumerate(
            #     pbar
            # ):
            #     batch_data = batch_data_process(batch)
            #     x, lmotion, music, wavnames = batch_data["fmotion"], batch_data["lmotion"], batch_data["music"], batch_data["wavnames"]
            #     x = torch.stack([x, lmotion], dim=1)
            #     total_loss, (loss, v_loss, foot_loss) = self.diffusion( 
            #         x, lmotion, music, t_override=None 
            #     )
            #     loss_dict = {
            #         "loss": loss.item(),
            #         "v_loss": v_loss.item(),
            #         "foot_loss": foot_loss.item(),
            #     }
                
            #     if self.accelerator.is_main_process:
            #         for key in loss_dict.keys():
            #             loss_dict[key] = float(loss_dict[key])

            #         loss_dict.update({'step': step + 1})
            #         pbar.set_postfix(loss_dict)

            #     self.optim.zero_grad()
            #     self.accelerator.backward(total_loss)
            #     self.optim.step()

            #     # ema update and train loss update only on main
            #     if self.accelerator.is_main_process:
            #         avg_loss += loss.detach().cpu().numpy()
            #         avg_vloss += v_loss.detach().cpu().numpy()
            #         avg_footloss += foot_loss.detach().cpu().numpy()
            #         if step % opt.ema_interval == 0:
            #             self.diffusion.ema.update_model_average(
            #                 self.diffusion.master_model, self.diffusion.model
            #             )
                        
            # Save model, log info, visualization for testing(from val dataset)
            if (epoch % opt.save_interval) == 0:
                # everyone waits here for the val loop to finish ( don't start next train epoch early)
                self.accelerator.wait_for_everyone()
                # save only if on main thread
                if self.accelerator.is_main_process:
                    self.eval()
                    # log
                    avg_loss /= len(train_data_loader)
                    avg_vloss /= len(train_data_loader)
                    avg_footloss /= len(train_data_loader)
                    log_dict = {
                        "Train Loss": avg_loss,
                        "V Loss": avg_vloss,
                        "Foot Loss": avg_footloss,
                    }
                    # wandb.log(log_dict)
                    self.writer.add_scalars("Train Loss", log_dict, epoch)
                    print(log_dict)
                    ckpt = {
                        "ema_state_dict": self.diffusion.master_model.state_dict(),
                        "model_state_dict": self.accelerator.unwrap_model(
                            self.model
                        ).state_dict(),
                        "optimizer_state_dict": self.optim.state_dict(),
                        "normalizer": self.normalizer,
                    }
                    torch.save(ckpt, os.path.join(wdir, f"train-{epoch}.pt")) # save ckpt
                    # generate a sample
                    render_count = 2
                    shape = (render_count, self.horizon*self.required_dancer_num, self.repr_dim)
                    print("Generating Sample")
                    # draw a cond from the test dataset
                    batch_data = batch_data_process(next(iter(test_data_loader)))
                    x, lmotion, music, wavnames = batch_data["fmotion"], batch_data["lmotion"], batch_data["music"], batch_data["wavnames"]
                    x = torch.stack([x, lmotion], dim=1)
                    
                    x = x.to(self.accelerator.device) 
                    x_traj_xy = x[:,0,:,[4,4+1]] # [*, 150, 2]
                    l_traj_xy = lmotion[:,:,[4,4+1]] # [*, 150, 2]
                    bs, seq, c = x_traj_xy.shape
                    x_traj = torch.zeros(bs, 2, seq, 3).to(x_traj_xy) # Note: Due to some historical baggage, we kept the option to input full xyz coordinates...
                    x_traj[:,0,:,[0,1]] = x_traj_xy[:,:,[0,1]] 
                    x_traj[:,1,:,[0,1]] = l_traj_xy[:,:,[0,1]] 

                    music = music.to(self.accelerator.device) # [*, 301, 438]
                    lmotion = lmotion.to(self.accelerator.device)

                    self.diffusion.render_sample( 
                        shape,
                        lmotion,
                        music,
                        self.normalizer,
                        epoch,
                        os.path.join(opt.render_dir, "train_" + opt.exp_name),
                        name=wavnames,
                        sound=True,
                        required_dancer_num = self.required_dancer_num,
                        x_0 = None, 
                    )
                    print(f"[MODEL SAVED at Epoch {epoch}]")
        
        if self.accelerator.is_main_process:
            # wandb.run.finish()
            self.writer.close()


    def given_trajectory_generation_loop(self, opt, train_dataloader, test_dataloader): 
        train_data_loader = self.accelerator.prepare(train_dataloader)
        test_data_loader = self.accelerator.prepare(test_dataloader)
        self.normalizer = train_dataloader.dataset.normalizer
        
        # boot up multi-gpu training. test dataloader is only on main process
        load_loop = (
            partial(tqdm, position=1, desc="Batch"),
            lambda x: x
        )

        render_count = 30 
        shape = (render_count, self.horizon*self.required_dancer_num, self.repr_dim)
        
        print("Begin validation with given trajectories")
        self.eval()
        for epoch in range(1, opt.epochs + 1):
            # draw a cond from the test dataset
            batch_data = batch_data_process(next(iter(train_data_loader)))
            x, lmotion, music, wavnames = batch_data["fmotion"], batch_data["lmotion"], batch_data["music"], batch_data["wavnames"]
            print("Generating Sample")
            x = x.cuda()
            x_traj_xy = x[:,:,[4,4+1]] # [*, 150, 2]
            l_traj_xy = lmotion[:,:,[4,4+1]] # [*, 150, 2]
            bs, seq, c = x_traj_xy.shape
            x_traj = torch.zeros(bs, 2, seq, 3).to(x_traj_xy) # Note: Due to some historical baggage, we kept the option to input full xyz coordinates...
            x_traj[:,0,:,[0,1]] = x_traj_xy[:,:,[0,1]] 
            x_traj[:,1,:,[0,1]] = l_traj_xy[:,:,[0,1]] 
            music = music.to(x) # [*, 301, 438]
            lmotion = lmotion.to(x)

            self.diffusion.render_sample( 
                shape,
                music[:render_count],
                lmotion[:render_count],
                self.normalizer,
                epoch,
                os.path.join(opt.render_dir, "Given_Train_" + opt.exp_name),
                name=wavnames[:render_count],
                sound=True,
                required_dancer_num= self.required_dancer_num,
                x_0 = x_traj[:render_count].permute(0,2,1,3).reshape(render_count,shape[1], 3), 
            )
            print(f"[TRAIN-RENDER SAVED at Epoch {epoch}]")

            
            shape = (render_count, self.horizon*self.required_dancer_num, self.repr_dim)
            print("Generating Sample")
            # draw a cond from the test dataset
            batch_data = batch_data_process(next(iter(test_data_loader)))
            x, lmotion, music, wavnames = batch_data["fmotion"], batch_data["lmotion"], batch_data["music"], batch_data["wavnames"]

            x = x.cuda() # [*, dn, 150, 70] [bs, 3, 150, 70]
            x_traj_xy = x[:,:,[4,4+1]] # [*, 150, 2]
            l_traj_xy = lmotion[:,:,[4,4+1]] # [*, 150, 2]
            bs, seq, c = x_traj_xy.shape
            x_traj = torch.zeros(bs, 2, seq, 3).to(x_traj_xy) # Note: Due to some historical baggage, we kept the option to input full xyz coordinates...
            x_traj[:,0,:,[0,1]] = x_traj_xy[:,:,[0,1]] 
            x_traj[:,1,:,[0,1]] = l_traj_xy[:,:,[0,1]] 

            music = music.to(x) # [*, 301, 438]
            lmotion = lmotion.to(x)
            self.diffusion.render_sample( 
                shape,
                music[:render_count],
                lmotion[:render_count],
                self.normalizer,
                epoch,
                os.path.join(opt.render_dir, "Given_Test_" + opt.exp_name),
                name=wavnames[:render_count],
                sound=True,
                required_dancer_num= self.required_dancer_num,
                x_0 = x_traj[:render_count].permute(0,2,1,3).reshape(render_count,shape[1], 3), # [2, seq, 3, 2] 
            )
            print(f"[VAL-RENDER SAVED at Epoch {epoch}]")


    def test_loop(self, opt): 
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
            train_dataset = pickle.load(open(train_tensor_dataset_path, "rb"))
            test_dataset = pickle.load(open(test_tensor_dataset_path, "rb"))
        else:
            train_dataset = AIOZDataset(
                data_path=opt.data_path,
                backup_path=opt.processed_data_dir,
                train=True,
                force_reload=opt.force_reload,
                required_dancer_num = self.required_dancer_num, 
                split_file = self.split_file,
            )
            test_dataset = AIOZDataset(
                data_path=opt.data_path,
                backup_path=opt.processed_data_dir,
                train=False,
                normalizer=train_dataset.normalizer,
                force_reload=opt.force_reload,
                required_dancer_num = self.required_dancer_num, 
                split_file = self.split_file,
            )
            # cache the dataset in case
            pickle.dump(train_dataset, open(train_tensor_dataset_path, "wb"))
            pickle.dump(test_dataset, open(test_tensor_dataset_path, "wb"))

        # set normalizer
        self.normalizer = test_dataset.normalizer

        # data loaders
        # decide number of workers based on cpu count
        num_cpus = multiprocessing.cpu_count()
        train_data_loader = DataLoader(
            train_dataset,
            batch_size=opt.batch_size,
            shuffle=True,
            num_workers=min(int(num_cpus * 0.75), 32),
            pin_memory=True,
            drop_last=True,
        )
        test_data_loader = DataLoader(
            test_dataset,
            batch_size=opt.batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True,
            drop_last=True,
        )

        # boot up multi-gpu training. test dataloader is only on main process
        load_loop = (
            partial(tqdm, position=1, desc="Batch"),
            lambda x: x
        )

        render_count = 30 
        shape = (render_count, self.horizon*self.required_dancer_num, self.repr_dim)

        ## init Trajectory Model
        trjm_args = option_traj.get_args_parser()
        torch.manual_seed(trjm_args.seed)
        window_size = trjm_args.window_size # align with training
        step = trjm_args.step
        traj_model = TrajDecoder(nfeats = trjm_args.nfeats, 
                  trans_layer = trjm_args.trans_layer, 
                  window_size = trjm_args.window_size,
                  ) 
        if trjm_args.checkpoint is not None:
            ckpt = torch.load(opt.traj_checkpoint, map_location='cpu')
            traj_model.load_state_dict(ckpt['net'], strict=True) 
            print('loading checkpoint from {}'.format(opt.traj_checkpoint))
        traj_model.cuda().eval()
        
        print("Begin testing with generated trajectories")
        self.eval()
        for epoch in range(1, opt.epochs + 1):
            # draw a cond from the test dataset
            batch_data = batch_data_process(next(iter(train_data_loader)))
            x, lmotion, music, wavnames = batch_data["fmotion"], batch_data["lmotion"], batch_data["music"], batch_data["wavnames"]
            x = torch.stack([x, lmotion], dim=1)
            print("Generating Sample")
            x = x.cuda()
            lmotion = lmotion.to(x)
            music = music.to(x) 

            # Autoregressively generate the full trajectory sequence
            pre_list = []

            # Extract initial xy trajectory from input data
            x_traj_xy = x[:,:,[4,4+1]] # [*, 150, 2]
            l_traj_xy = lmotion[:,:,[4,4+1]] # [*, 150, 2]
            bs, seq, c = x_traj_xy.shape
            x_traj = torch.zeros(bs, 2, seq, 3).to(x_traj_xy) # Note: Due to some historical baggage, we kept the option to input full xyz coordinates...
            x_traj[:,0,:,[0,1]] = x_traj_xy[:,:,[0,1]] 
            x_traj[:,1,:,[0,1]] = l_traj_xy[:,:,[0,1]] 

            # Initialize the first window for trajectory prediction
            cond_traj = x_traj[:, :,:window_size,[0,1]] 
            pre_list.append(cond_traj) 
            cond_len = music.shape[1]

            # Slide a window over the cond features
            # cond sequence length is (window_size + step) * 2 because cond FPS is twice the motion FPS
            # Hence, move the cond window by step*2 each time
            for start in range(0, cond_len + 1-(window_size+step)*2, step*2):  
                # Predict the next trajectory segment
                pre_traj = traj_model(cond_traj, music[ :, start:start + (window_size+step) * 2], lmotion[ :, start:start + (window_size+step) * 2]) 
                cond_traj = pre_traj
                pre_list.append(pre_traj[:,:,-step:])
            
            # Concatenate all trajectory segments into a single sequence
            x_traj = torch.cat(pre_list,dim = 2) 

            # Optional: process trajectory with smoothing or constraints
            x_traj = kalman_smooth_batch(x_traj.cpu().detach().numpy())
            x_traj = torch.from_numpy(x_traj).to(dtype=x.dtype, device=x.device)
            
            # Pad the trajectory to 3D space by adding a zero z-coordinate
            bs, dn, seq, c = x_traj.shape
            x_traj_padding = torch.zeros(bs, dn, seq, 3).to(x_traj)
            x_traj_padding[:,:,:,[0,1]] = x_traj[:,:,:,[0,1]] 

            self.diffusion.render_sample( 
                shape,
                music[:render_count],
                lmotion[:render_count],
                self.normalizer,
                epoch,
                os.path.join(opt.render_dir, "TRAIN_" + opt.exp_name),
                name=wavnames[:render_count],
                sound=True,
                required_dancer_num= self.required_dancer_num,
                x_0 = x_traj_padding[:render_count].permute(0,2,1,3).reshape(render_count,shape[1], 3), 
            )
            print(f"[TRAIN-RENDER SAVED at Epoch {epoch}]")

            
            shape = (render_count, self.horizon*self.required_dancer_num, self.repr_dim)
            print("Generating Sample")
            # draw a cond from the test dataset
            batch_data = batch_data_process(next(iter(test_data_loader)))
            x, lmotion, music, wavnames = batch_data["fmotion"], batch_data["lmotion"], batch_data["music"], batch_data["wavnames"]
            x = torch.stack([x, lmotion], dim=1)
            x = x.cuda() 
            lmotion = lmotion.cuda()
            music = music.cuda()

            # Autoregressively generate the full trajectory sequence
            pre_list = []

            # Extract initial xy trajectory from input data
            x_traj_xy = x[:,:,[4,4+1]] # [*, 150, 2]
            l_traj_xy = lmotion[:,:,[4,4+1]] # [*, 150, 2]
            bs, seq, c = x_traj_xy.shape
            x_traj = torch.zeros(bs, 2, seq, 3).to(x_traj_xy) # Note: Due to some historical baggage, we kept the option to input full xyz coordinates...
            x_traj[:,0,:,[0,1]] = x_traj_xy[:,:,[0,1]] 
            x_traj[:,1,:,[0,1]] = l_traj_xy[:,:,[0,1]] 

            # Initialize the first window for trajectory prediction
            cond_traj = x_traj[:, :,:window_size,[0,1]] 
            pre_list.append(cond_traj) 
            cond_len = music.shape[1]

            # Slide a window over the cond features
            # cond sequence length is (window_size + step) * 2 because cond FPS is twice the motion FPS
            # Hence, move the cond window by step*2 each time
            for start in range(0, cond_len + 1-(window_size+step)*2, step*2):  
                # Predict the next trajectory segment
                pre_traj = traj_model(cond_traj, music[ :, start:start + (window_size+step) * 2], lmotion[ :, start:start + (window_size+step) * 2]) 
                cond_traj = pre_traj
                pre_list.append(pre_traj[:,:,-step:])
            
            # Concatenate all trajectory segments into a single sequence
            x_traj = torch.cat(pre_list,dim = 2) 

            # Optional: process trajectory with smoothing or constraints
            x_traj = kalman_smooth_batch(x_traj.cpu().detach().numpy())
            x_traj = torch.from_numpy(x_traj).to(dtype=x.dtype, device=x.device)

            # Pad the trajectory to 3D space by adding a zero z-coordinate
            bs, dn, seq, c = x_traj.shape
            x_traj_padding = torch.zeros(bs, dn, seq, 3).to(x_traj)
            x_traj_padding[:,:,:,[0,1]] = x_traj[:,:,:,[0,1]] 

            self.diffusion.render_sample( 
                shape,
                music[:render_count],
                lmotion[:render_count],
                self.normalizer,
                epoch,
                os.path.join(opt.render_dir, "TEST_" + opt.exp_name),
                name=wavnames[:render_count],
                sound=True,
                required_dancer_num= self.required_dancer_num,
                x_0 = x_traj_padding[:render_count].permute(0,2,1,3).reshape(render_count,shape[1], 3), # [2, seq, 3, 2] 
            )
            print(f"[TEST-RENDER SAVED at Epoch {epoch}]")



    def render_sample( # Renders long motion sequences for testing or visualization.
        self, data_tuple, label, render_dir, render_count=-1, render=True, x_0 = None, render_len = 512
    ):
        _, music, lmotion, wavname = data_tuple
        assert len(music.shape) == 3
        # Automatically determine the number of audio segments to render
        if render_count < 0: 
            render_count = len(music)
        # Define the shape of the output motion sequence:
        #   - batch size: render_count
        #   - sequence length: horizon * number of dancers
        #   - feature dimension: representation dimension per frame
        shape = (render_count, self.horizon*self.required_dancer_num, self.repr_dim)
        music = music.to(self.accelerator.device)
        lmotion = lmotion.to(self.accelerator.device)
        self.diffusion.render_sample(
            shape,
            music[:render_count],
            lmotion[:render_count],
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
        )
