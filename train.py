from args import parse_train_opt
from TCDiff import TCDiff
import warnings
import pickle as pkl
warnings.filterwarnings('ignore')
import os
import codecs as cs

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def train(opt):
    # train_dataloader = get_dataset_loader(batch_size=opt.batch_size, split="train", num_workers=8, full_length=False)
    # test_dataloader = get_dataset_loader(batch_size=1, split="test", num_workers=8, full_length=False)
    
    # if opt.use_normalizer:
    #     print("Using data normalizer")
    #     dataset = train_dataloader.dataset
    #     datapath = dataset.datapath
    #     if not os.path.exists(os.path.join(datapath, "normalizer.pkl")):
    #         from dataset.preprocess import preprocess_data
    #         preprocess_data(dataset, split="train")
    #     normalizer = pkl.load(
    #         open(os.path.join(datapath, "normalizer.pkl"),"rb")
    #     )
    #     setattr(train_dataloader.dataset, "normalizer", normalizer)


    ckpt_epoch = opt.checkpoint.split("-")[-1].split(".")[0]
    model = TCDiff(checkpoint_path = opt.checkpoint, learning_rate=opt.learning_rate, \
        window_size=opt.window_size, required_dancer_num = opt.required_dancer_num, opt=opt)
    if opt.mode == "train":
        model.train_loop(opt)
    elif opt.mode == "render":
        # model.long_sample(ckpt_epoch, opt.render_dir, 240)
        # model.long_sample(ckpt_epoch, opt.render_dir, 150)
        model.long_sample(ckpt_epoch, opt.render_dir, None)
    elif opt.mode == "val_without_TrajModel":
        model.given_trajectory_generation_loop(opt)
    elif opt.mode == "test":
        model.test_loop(opt)
    else:
        raise ValueError(f"Invalid mode: {opt.mode}. Must be one of ['train', 'val_without_TrajModel', 'test'].")

if __name__ == "__main__":
    opt = parse_train_opt()
    train(opt)
