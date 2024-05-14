import denoising_diffusion_pytorch
from denoising_diffusion_pytorch import GaussianDiffusion, Trainer, Unet
import PIL as Image
import torch
from torchvision import utils
import os

from accelerate import Accelerator

#@torch.no_grad()
#checkpoint = torch.load('/workspace/results/model-1.pt')

def exists(x):
    return x is not None

def produce_sample(model, model_dir="/workspace/results/model-sweep-best.pt", num_imgs_gen=16, multiple_in_one=True, imgs_in_one=16, save_dir=''):
    data = torch.load(str(model_dir), map_location='cpu')

    accelerator = Accelerator()

    model = accelerator.unwrap_model(model).cuda()
    model.load_state_dict(data['model'])

    for j in range(0, num_imgs_gen):
        save_path=os.path.join(save_dir, f'sample{j}.png')
        if multiple_in_one:
            samples = list(GaussianDiffusion.sample(model, batch_size=imgs_in_one))
            utils.save_image(samples, save_path, nrow =np.sqrt(imgs_in_one))
        else:
            samples = list(GaussianDiffusion.sample(model, batch_size=1))
            utils.save_image(samples, save_path)

if __name__=='__main__':
    model = Unet(
        dim=64,
        dim_mults = (1, 2, 4, 8),
        flash_attn = True,
        channels=1
    )

    diffusion = GaussianDiffusion(
        model,
        image_size = 256,
        timesteps = 1000,           # number of steps
        sampling_timesteps = 250,    # number of sampling timesteps (using ddim for faster inference [see citation for ddim paper])
        objective = 'pred_x0',
    ).cuda()

    # trainer = Trainer(
    #             diffusion,
    #             '/workspace/structure_images_contrast_10000/',
    #             train_batch_size=8, #=config.train_batch_size,
    #             train_num_steps=10000, #=config.train_num_steps,
    #             train_lr = 8e-5,
    #             gradient_accumulate_every = 2,
    #             ema_decay = 0.9,
    #             amp = True,
    #             calculate_fid = True
    # )

    produce_sample(
        diffusion,
        # model_dir="/media/ps2/leuthold/denoising-diffusion-pytorch/results/model-sweep-best.pt",
        model_dir="/media/ps2/leuthold/denoising-diffusion-pytorch/results_freeform/model-freeform_first-best.pt",
        num_imgs_gen=10,
        multiple_in_one=False,
        imgs_in_one=16,          # only if all_in_one=True, has to be the square of an int!
        save_dir='./',
    )