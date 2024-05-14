# usage: python train_controlnet_rectangle.py ./configs/args_test.yaml

import torch
from torch.utils.data import Dataset, DataLoader, random_split
from farfield_dataset import MyDataset
from learners_controlled_rectangle_large import CombinedModel
import sys
sys.path.append("../")
import argparse
import yaml
import itertools
from tqdm import tqdm
import wandb
from torchvision import utils, transforms
from train_util import *
import numpy as np
from numpy import random
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt


def plot_time_steps(imgs, name):
    plt.figure(figsize=(3, len(imgs)*1))
    for i in range(len(imgs)):
        plt.subplot(len(imgs), 1, i+1)
        plt.imshow(imgs[i][0])
        plt.colorbar()
        plt.xticks([])
        plt.yticks([])
    plt.savefig(name, transparent=True, dpi=300, bbox_inches='tight', pad_inches=0)

def angle_distance(ang1, ang2):
	diff = ang1-ang2
	return min(abs(diff), abs(diff+360), abs(diff-360))


def inverse_farfield(angle, phase_deg, magnitude=1.5, num_peaks=2, half_spread=20, N_theta=360):
	farfield_real_total = torch.zeros(N_theta)
	farfield_imag_toatl = torch.zeros(N_theta)

	for j in range(num_peaks):
		farfield_real = torch.zeros(N_theta)
		farfield_imag = torch.zeros(N_theta)

		for i in range(N_theta):
			farfield_real[i] = 0 if angle_distance(i, angle[j]) > half_spread else magnitude*np.cos(phase_deg*np.pi/180)*(np.cos((i-angle[j])/half_spread*np.pi)+1)
			farfield_imag[i] = 0 if angle_distance(i, angle[j]) > half_spread else magnitude*np.sin(phase_deg*np.pi/180)*(np.cos((i-angle[j])/half_spread*np.pi)+1)
		farfield_real_total += farfield_real
		farfield_imag_toatl += farfield_imag

	return torch.stack((farfield_real_total, farfield_imag_toatl), dim=-1)[None]

def create_dataset(args, magnitude):
	data = np.empty((args.N_data, 360))
	angles = random.randint(args.min_angle, args.max_angle, size=(args.N_data, args.num_peaks), dtype = int)
	sign = random.choice([-1, 1], size=(args.N_data, args.num_peaks))
	angles = sign*angles
	phase_deg = random.randint(0, 360, size=(args.N_data), dtype = int)

	for i in range(args.N_data):
		torch_data = inverse_farfield(angles[i], phase_deg[i], magnitude=magnitude, num_peaks=args.num_peaks, half_spread=args.half_spread)[:,:,0].numpy() + 1j*inverse_farfield(angles[i], phase_deg[i], magnitude=magnitude, num_peaks=args.num_peaks, half_spread=args.half_spread)[:,:,1].numpy()
		data[i] = torch_data.squeeze()

	return data

def visualize_structure(struct, step, index_1, index_2 = None, save_name = '', model_name=''):
	for i in range(index_1):
		if index_2 is not None:
			for j in range(index_2):

				img = Image.fromarray((struct[i][j][0]*255).astype(np.uint8), 'L')
				img.save(save_name + '/' + f'/timesteps/1{model_name}_sample_No{step}_{i}th_img_{(j)*(1000/(index_2-1))}th_timestep.png')
		else:
			img = Image.fromarray((struct[i][0]*255).astype(np.uint8), 'L')
			img.save(save_name + '/' + f'/timesteps/1{model_name}_sample_No{step}_{i}th_img.png')				
		


def main(args):
	device = torch.device('cuda')
	torch.cuda.set_device(args.cuda)

	# Configs
	resume_path = args.model_saving_path + '/models/' + args.model_name + ".pt"
	model = CombinedModel(args)

	# load the state_dict from a new CombinedModel with trained diffusion weights and empty controlnet weights
	# which is produced by tool_add_control.py
	model.load_state_dict(torch.load(resume_path)['state_dict'])
	model = model.cuda()
	print(f"model weights loaded from {resume_path}")

	# First use cpu to load models. Pytorch Lightning will automatically move it to GPUs.
	model.learning_rate = args.start_lr
	model.diffusion_locked = args.diffusion_locked
	model.only_mid_control = args.only_mid_control

	# dataset
	original_eps = torch.from_numpy(np.load(args.data_folder + '/' + 'input_eps.npy', mmap_mode='r').astype(np.float32))
	original_eps = torch.unsqueeze(original_eps, dim=1).cuda()

	original_farfields = torch.from_numpy(np.load(args.data_folder + '/' + 'Far_Ey.npy', mmap_mode='r').astype(np.complex64))
	original_farfields = original_farfields[:,:-1].cuda()
	mean_real = torch.max(original_farfields.type(torch.complex64).real, dim=1)[0].mean().item()
	mean_imag = torch.max(original_farfields.type(torch.complex64).imag, dim=1)[0].mean()
	magnitude = torch.sqrt(mean_real**2 + mean_imag**2).item()
	print("magnitude is: ", magnitude)
	original_farfields = process_farfield(original_farfields)



	if not args.handcrafted:
		test_farfields = torch.from_numpy(np.load(args.data_folder + '/' + 'Far_Ey.npy', mmap_mode='r').astype(np.complex64)).cuda()
		test_input_eps = torch.from_numpy(np.load(args.data_folder + '/' + 'input_eps.npy', mmap_mode='r'))
		if test_farfields.shape[1] == 361:
			test_farfields = test_farfields[:,:-1]
		test_farfields = process_farfield(test_farfields)

		shuffled_indices = np.random.permutation(test_input_eps.shape[0])
		test_farfields = test_farfields[shuffled_indices]
		test_input_eps = test_input_eps[shuffled_indices]
		test_farfields = test_farfields[0:args.N_data]
		test_input_eps = test_input_eps[0:args.N_data]
		assert test_farfields.shape[0] == args.N_data
	else:
		test_farfields = torch.from_numpy(create_dataset(args, magnitude).astype(np.complex64)).cuda()
		test_farfields = process_farfield(test_farfields)

	if args.interpolate:
		df = pd.DataFrame(columns = [
			'Step', 
			'Max farfield sim ori vs. in', 
			'Average farfield sim. out vs. in',
			'Max farfield sim out vs. in',
			'Index of max farfield sim out vs. in',
			'Max structure sim. ori vs. in',
			'Average structure sim. out vs. in',
			'Positions of best generated structure', 
			'Shape of best generated structure',
			'Positions of best original structure', 
			'Shape of best original structure']
		)

	
	steps = test_farfields.shape[0]
	assert steps == args.N_data

	if args.return_all_timesteps:
		test_generated_samples = np.empty((steps, args.num_parallel_samples, (args.sampling_timesteps//10)+1, 1, args.image_sizex, args.image_sizey))
	else:
		test_generated_samples = np.empty((steps, args.num_parallel_samples, 1, args.image_sizex, args.image_sizey))

	# sampling loop:
	with torch.no_grad():
		step = 0
		similarity_scores = np.empty((args.N_data, args.top_n))
		similarity_average_list = []
		with tqdm(initial = step, total = steps) as pbar:
			sample_batch_size = args.num_parallel_samples
			while step < steps:

				test_farfield = test_farfields[step]
				if not args.handcrafted:
					test_eps = test_input_eps[step]

				
				test_sample = model.sample(test_farfield, batch_size = sample_batch_size, return_all_timesteps = args.return_all_timesteps, return_every_n_timestep=args.return_every_n_timestep)
				for j in range(args.num_parallel_samples):
					test_generated_samples[step][j] = np.array(test_sample[j].cpu(), dtype = np.float32)
			
				if args.interpolate:
					print("in interpolate")
					if not args.handcrafted:
						max_sim_farfield_in_ori, average_sim_farfield_in_out, max_sim_farfield_in_out, max_sim_farfield_index, max_sim_struct_in_ori, average_sim_struct_in_out, best_generated_struct_positions, best_generated_struct_shape, best_original_struct_positions, best_original_struct_shape = test_interpolation(step, args, test_generated_samples[step], test_farfield.cpu().numpy(), test_input_eps.cpu().numpy(), original_eps.cpu().numpy(), original_farfields.cpu().numpy(), args.num_devices_for_comparison, handcrafted = args.handcrafted, only_best_in_one_plot=args.only_best_in_one_plot, component = 'Ey')
					else:
						max_sim_farfield_in_ori, average_sim_farfield_in_out, max_sim_farfield_in_out, max_sim_farfield_index, best_generated_struct_positions, best_generated_struct_shape, best_original_struct_positions, best_original_struct_shape = test_interpolation(step, args, test_generated_samples[step], test_farfield.cpu().numpy(), None, original_eps.cpu().numpy(), original_farfields.cpu().numpy(), args.num_devices_for_comparison, handcrafted = args.handcrafted, only_best_in_one_plot=args.only_best_in_one_plot, component = 'Ey')
						max_sim_struct_in_ori, average_sim_struct_in_out = 0, 0
					
					print(f'Step: {step}: Max_sim_farfield_in_ori: {max_sim_farfield_in_ori}')
					print(f'average_sim_farfield_in_out: {average_sim_farfield_in_out}')
					print(f'Max_sim_farfield_in_out: {max_sim_farfield_in_out}')
					print(f'index_of_max_sim_in_out: {max_sim_farfield_index}')
					print(f'max_sim_struct_in_ori: {max_sim_struct_in_ori}')
					print(f'average_sim_struct_in_out: {average_sim_struct_in_out}')
					df.loc[step] = [step, 
									max_sim_farfield_in_ori, 
									average_sim_farfield_in_out, 
									max_sim_farfield_in_out, 
									max_sim_farfield_index, 
									max_sim_struct_in_ori, 
									average_sim_struct_in_out,
									best_generated_struct_positions, 
									best_generated_struct_shape,
									best_original_struct_positions, 
									best_original_struct_shape]
				
				
				if args.analyse_w_original:
					generated_farfields = np.empty((test_generated_samples[step].shape[0], test_generated_samples[step].shape[1], 360, 2))
					generated_nearfields = np.empty((test_generated_samples[step].shape[0], test_generated_samples[step].shape[1], 2, 250, 250))
					theta_obs = 0
					
					for j in range(args.num_parallel_samples):
						generated_farfields[j], generated_nearfields[j], theta_obs = get_farfield(args, test_generated_samples[step][j])
					test_eps=test_eps.unsqueeze(0)
					input_farfield, input_nearfield, theta_obs = get_farfield(args, test_eps)
					
					similarity_score, generated_farfield, index = evaluate_top_n(test_farfield.cpu().numpy(), generated_farfields, simmilarity_score, top_n=args.top_n)
					generated_farfield_debug = generated_farfields[index]
					similarity_scores[step] = similarity_score
					generated_nearfield = generated_nearfields[index]
					best_test_sample = test_generated_samples[step][index]
					generated_nearfield = generated_nearfield[:,:,0,:,:] + 1j*generated_nearfield[:,:,1,:,:]
					generated_nearfield = generated_nearfield.squeeze(1)

					plt.figure(figsize=(20, 5*(1+args.top_n)), dpi=700)
					plot_row(test_eps.squeeze().T, input_nearfield.T, theta_obs, test_farfield.cpu().numpy(), 1+args.top_n, 1, color = 'orange', vm_real=None, vm_imag=None, vm_intensity=None, with_power=True)
					for j in range(args.top_n):
						plot_row(best_test_sample[j].squeeze().T, generated_nearfield[j].T, theta_obs, generated_farfield[j].squeeze(), 1+args.top_n, 2+j, color = 'dodgerblue', vm_real=None, vm_imag=None, vm_intensity=None, with_power=True)
					plt.tight_layout()
					plt.savefig(f'./large_model/samples/thesis/n_{args.n_mat}_original_dataset_analysis_{step}.png', transparent=True, )


				if args.inverse_design:
					generated_farfields = np.empty((test_generated_samples[step].shape[0], test_generated_samples[step].shape[1], 360, 2))
					generated_nearfields = np.empty((test_generated_samples[step].shape[0], test_generated_samples[step].shape[1], 2, 250, 250))
					theta_obs = 0
					
					for i in range(args.num_parallel_samples):
						generated_farfields[i], generated_nearfields[i], theta_obs = get_farfield(args, test_generated_samples[step][i])
					similarity_score, generated_farfield, index = evaluate_top_n(test_farfield.cpu().numpy(), generated_farfields, simmilarity_score, top_n=args.top_n)
					similarity_scores[step] = similarity_score
					generated_nearfield = generated_nearfields[index]
					best_test_sample = test_generated_samples[step][index]
					generated_nearfield = generated_nearfield[:,:,0,:,:] + 1j*generated_nearfield[:,:,1,:,:]

					plt.figure(figsize=(20, 5*(1+args.top_n)), dpi=400)
					plot_row(None, None, theta_obs, test_farfield.cpu().numpy(), 1, 1, color = 'dodgerblue', vm_real=None, vm_imag=None, vm_intensity=None, with_power=True)
					for j in range(args.top_n):
						plot_row(best_test_sample[j].squeeze().T, generated_nearfield[j].T, theta_obs, generated_farfield[j].squeeze(), 1, 1, color = 'yellowgreen', vm_real=None, vm_imag=None, vm_intensity=None, with_power=True)
					plt.tight_layout()
					plt.savefig(f'./inverse_design/spread/inverse_design_pixel_{args.half_spread}_angle_{step}.png', transparent=True, )
				
				print(f'Similarity score at step {step}:', similarity_score)

				similarity_average_list.append(np.mean(similarity_scores[:step]))

				step += 1
				pbar.update(1)

			if args.return_all_timesteps:
				plot_time_steps(test_generated_samples[0][0], './large_model/samples/structures/timesteps/0test_timesteps.png')

	if args.interpolate:
		print(df)
		df.to_csv('data_interpolation_32.csv', index=False)

	print(f'similarity_score average for {args.N_data} devices: ', np.mean(similarity_scores))
	print(f'standard deviation: ', np.std(similarity_scores))
	print(f'all similarity_scores: ', similarity_scores)
	
	np.save(args.save_dir_samples + '/' + 'structures' + '/' + 'generated_structures_inverse_design.npy', test_generated_samples)



if __name__ == '__main__':
	# usage: python3 interpolation_controlnet_rectangle.py ./large_model/configs/args_test_interpolation.yaml
	config_path = sys.argv[1]

	parser = argparse.ArgumentParser(description="Arguments for the controlnet model")
	with open(config_path, "r") as file:
		config = yaml.safe_load(file)

	# Update the parser's default values with the loaded configurations
	args = argparse.Namespace(**config)

	main(args)
