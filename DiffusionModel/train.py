from denoising_diffusion_pytorch_rectangle import Unet, GaussianDiffusion, Trainer
import torch
import wandb
import argparse

def training(args):
    model = Unet(
        dim=128,
        dim_mults = (1, 2, 4, 8),
        flash_attn = True
    )

    diffusion = GaussianDiffusion(
        model,
        image_size = (args.img_sx, args.img_sy),
        timesteps = 1000,           # number of steps
        sampling_timesteps = 100,    # number of sampling timesteps (using ddim for faster inference [see citation for ddim paper])
        objective = 'pred_x0',
    )


    device = torch.device('cuda')
    torch.cuda.set_device(args.cuda_device)
    
    trainer = Trainer(
        diffusion,
        args.input_dir,
        train_batch_size=args.train_batch_size,
        train_num_steps=args.train_num_steps,
        save_and_sample_every = 100,
        num_samples = 9,
        train_lr = 8e-5,
        gradient_accumulate_every = 2,
        ema_decay = args.ema_decay,
        amp = True,
        calculate_fid = True,
        num_fid_samples = 16,
        save_best_and_latest_only = True,
        save_name=args.save_name,
        results_folder = args.save_dir,
        log_wandb_every=50
    )
    
    trainer.train()



if __name__ == '__main__':

    argparser = argparse.ArgumentParser()

    argparser.add_argument("--input_dir", type=str, help="", default = "/media/ps2/chenkaim/data/controlnet/waveynet_1to6bars_MFS4_th_10_30/n_3.7_dL25")
    argparser.add_argument("--save_dir", type=str, help="", default = "/media/lts0/chenkaim/checkpoints/diffusion/waveynet_1to6bars_MFS4_th_10_30/")
    argparser.add_argument("--img_sx", type=int, help="", default = 32)
    argparser.add_argument("--img_sy", type=int, help="", default = 96)
    argparser.add_argument("--train_batch_size", type=int, help="", default = 16)
    argparser.add_argument("--train_num_steps", type=int, help="", default = 10000)


    argparser.add_argument("--save_name", type=str, help="", default = "simple_diff_10k")
    argparser.add_argument("--ema_decay", type=float, help="", default = 0.95)
    argparser.add_argument("--cuda_device", type=int, help="select the system GPU to use", default = 0)

    args = argparser.parse_args()

    training(args)

