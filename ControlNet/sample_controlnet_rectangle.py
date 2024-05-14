# usage: python train_controlnet.py ./configs/args_test.yaml

import torch
from torch.utils.data import DataLoader
from farfield_dataset import MyDataset
from learners_controlled_rectangle import CombinedModel
import sys
sys.path.append("../")
import argparse
import yaml
import itertools
from tqdm import tqdm
from PIL import Image
from torchvision import utils
import os
import math
import numpy as np

def main(args):
    # Configs
    resume_path = args.model_saving_path + '/' + args.model_name + ".ckpt"
    model = CombinedModel(args)

    # load the state_dict from a new CombinedModel with trained diffusion weights and empty controlnet weights
    # which is produced by tool_add_control.py
    model.load_state_dict(torch.load(resume_path))
    model = model.cuda()
    print(f"model weights loaded from {resume_path}")

    # First use CPU to load models. PyTorch Lightning will automatically move it to GPUs.
    model.learning_rate = args.start_lr
    model.diffusion_locked = args.diffusion_locked
    model.only_mid_control = args.only_mid_control

    # Misc
    dataset = MyDataset(args)
    dataloader = DataLoader(dataset, num_workers=0, batch_size=args.batch_size_sampling, shuffle=True)
    dl_iterator = itertools.cycle(dataloader)

    original_eps = np.empty([args.num_samples*args.batch_size_sampling, 1, args.image_sizex, args.image_sizey], dtype=np.float32)
    generated_eps = np.empty([args.num_samples*args.batch_size_sampling, args.num_parallel_samples, 1, args.image_sizex, args.image_sizey], dtype=np.float32)
    original_farfield = np.empty([args.num_samples*args.batch_size_sampling, 2, 360])

    # Sampling loop
    step = 0
    with tqdm(initial=step, total=args.num_samples) as pbar:
        while step < args.num_samples:
            with torch.inference_mode():
                save_path = os.path.join(args.save_dir_samples, f'structures/generated_corrected_{step}_2.png')
                save_path_original = os.path.join(args.save_dir_samples, f'structures/original_corrected_{step}_2.png')
                data = next(dl_iterator)
                # print("Shape of data: ", data.keys())           # dict_keys(['eps', 'far_Ex']) where data['eps'].shape=[1, 1, 96, 96] and data['far_Ex'].shape=[1, 2, 360]
                eps, far_Ex = data['eps'].cuda(), data['far_Ex'].cuda()

                samples = torch.empty(0, eps.size(1), eps.size(2), eps.size(3)).cuda()
                for i in range(args.num_parallel_samples):
                    sample = model.sample(far_Ex, batch_size = args.batch_size_sampling)
                    # print("Sample min: ", torch.max(sample), "Sample max: ", torch.min(sample))
                    assert sample.size(0) == args.batch_size_sampling
                    samples = torch.cat((samples, sample), dim=0)

                if step < 0:
                    utils.save_image(samples, save_path, nrow=np.sqrt(args.num_parallel_samples))
                    utils.save_image(eps, save_path_original)
                samples_array=np.array(samples.cpu(), dtype=np.float32)
                original_array=np.array(eps.cpu(), dtype=np.float32)
                far_Ex_array=np.array(far_Ex.cpu(), dtype=np.float32)
                

                for i in range(args.batch_size_sampling):
                    original_eps[step*args.batch_size_sampling+i]=original_array[i]
                    original_farfield[step*args.batch_size_sampling+i]=far_Ex_array[i]
                    for j in range(args.num_parallel_samples):
                        generated_eps[step*args.batch_size_sampling+i][j]=samples_array[j]

                step += 1
                pbar.update(1)

    np.save(args.save_dir_samples+f"data_files/original_eps_{args.save_name}.npy", original_eps)
    np.save(args.save_dir_samples+f"data_files/generated_eps_{args.save_name}.npy", generated_eps)
    np.save(args.save_dir_samples+f"data_files/original_farfield_{args.save_name}.npy", original_farfield)
    

            

if __name__ == '__main__':
	# usage: python3 sample_controlnet.py ./configs/args_for_testing_during_train.yaml
	config_path = sys.argv[1]

	parser = argparse.ArgumentParser(description="Arguments for the controlnet model")
	with open(config_path, "r") as file:
		config = yaml.safe_load(file)

	# Update the parser's default values with the loaded configurations
	args = argparse.Namespace(**config)
	main(args)
