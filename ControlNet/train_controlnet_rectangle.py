# usage: python train_controlnet_rectangle.py ./configs/args_test.yaml

import torch
from torch.utils.data import Dataset, DataLoader, random_split
from farfield_dataset import MyDataset
from learners_controlled_rectangle import CombinedModel
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



def main(args):
	device = torch.device('cuda')
	torch.cuda.set_device(args.cuda)

	# Configs
	resume_path = args.model_saving_path + '/models/' + args.model_name + ".pt"
	model = CombinedModel(args)

	# load the state_dict from a new CombinedModel with trained diffusion weights and empty controlnet weights
	# which is produced by tool_add_control.py
	model.load_state_dict(torch.load(resume_path)) #['state_dict'] for 100k dataset
	model = model.cuda()
	print(f"model weights loaded from {resume_path}")

	# First use cpu to load models. Pytorch Lightning will automatically move it to GPUs.
	model.learning_rate = args.start_lr
	model.diffusion_locked = args.diffusion_locked
	model.only_mid_control = args.only_mid_control

	# Misc
	dataset = MyDataset(args)

	train_size = int(0.8 * len(dataset))
	test_size = len(dataset) - train_size
	train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

	train_dataloader = DataLoader(train_dataset, num_workers=0, batch_size=args.batch_size, shuffle=True)
	test_dataloader = DataLoader(test_dataset, num_workers=0, batch_size=args.batch_size, shuffle=True)
	train_dl_iterator = itertools.cycle(train_dataloader)
	test_dl_iterator = itertools.cycle(test_dataloader)

	optimizer = model.build_optim()
	# training loop:

	step = 0
	running_mean_loss = 0
	total_loss_list = []
	running_loss_list = []
	test_loss_list = []
	similarity_score_list = []
	test_similarity_score_list = []
	sample_steps = []

	data_dict = {}
		
	with tqdm(initial = step, total = args.train_num_steps) as pbar:
		sample_batch_size = 1
		generated_eps = np.empty(((int(args.train_num_steps/5)), sample_batch_size, args.num_parallel_samples, 1, 96, 96))
		original_eps = np.empty(((int(args.train_num_steps/5)), sample_batch_size, 1, 96, 96))
		while step < args.train_num_steps:
	
			optimizer.zero_grad()
			
			total_loss = 0.
			for _ in range(args.gradient_accumulate_every):
				data = next(train_dl_iterator)
				eps, farfield, farfield_mean = data['eps'].cuda(), data['farfield'].cuda(), data['farfield_mean'].cuda()
				farfield = process_farfield(farfield)
				if args.use_nearfield:
					physical_hint = nearfield
				else:
					physical_hint = farfield

				loss = model(eps, physical_hint)
				total_loss += loss / args.gradient_accumulate_every

				with torch.no_grad():
					if step != 0 and not (step % args.sample_every):
						test_data = next(test_dl_iterator)
						test_eps, test_farfield, test_farfield_mean = test_data['eps'].cuda(), test_data['farfield'].cuda(), test_data['farfield_mean'].cuda()
						test_farfield = process_farfield(test_farfield)
						test_farfield_mean = process_farfield(test_farfield_mean)
						if args.use_nearfield:
							test_physical_hint = test_nearfield
						else:
							test_physical_hint = test_farfield
						
						test_loss = model(test_eps, test_physical_hint)
						
						test_generated_samples = np.empty((args.num_parallel_samples, 1, args.image_sizex, args.image_sizey))
						generated_samples = np.empty((args.num_parallel_samples, 1, args.image_sizex, args.image_sizey))


						for j in range(args.num_parallel_samples):
							sample = model.sample(physical_hint[0:sample_batch_size], batch_size = sample_batch_size)
							generated_samples[j] = np.array(sample.cpu(), dtype = np.float32)
							test_sample = model.sample(test_physical_hint[0:sample_batch_size], batch_size = sample_batch_size)
							test_generated_samples[j] = np.array(test_sample.cpu(), dtype = np.float32)
						
						similarity, image = analyze_one_sample_source_in(step, args, eps[0:sample_batch_size].cpu().numpy(), farfield[0:sample_batch_size].cpu().numpy(), generated_samples, component='Ey')
						original = np.array(eps[0:sample_batch_size].cpu(), dtype = np.float32)
						test_similarity, test_image = analyze_one_sample_source_in(step, args, test_eps[0:sample_batch_size].cpu().numpy(), test_farfield[0:sample_batch_size].cpu().numpy(), test_generated_samples, component='Ey')
						test_original = np.array(test_eps[0:sample_batch_size].cpu(), dtype = np.float32)

						test_similarity_score_list.append(test_similarity)
						similarity_score_list.append(similarity)
						sample_steps.append(step)
						test_loss_list.append(test_loss.item())

						wandb.log({
									"generated": [wandb.Image(image) for image in generated_samples],
									"test_generated": [wandb.Image(image) for image in test_generated_samples],
									"original": [wandb.Image(original_image) for original_image in original],
									"Train_Fields": wandb.Image(image),
									"Test_Fields": wandb.Image(test_image),
									"similarity_scores": {"similarity_score": similarity, "test_similarity_score": test_similarity}, 
									"similarity_score_train": similarity,
									"similairty_score_test": test_similarity,
									"test_loss": test_loss,
									"losses": {"loss": loss, "test_loss": test_loss}}
						)

			running_mean_loss = 0.95*running_mean_loss+0.05*total_loss.item()
			print("running_mean_loss: ", running_mean_loss)
			total_loss_list.append(total_loss.item())
			running_loss_list.append(running_mean_loss)

			plt.figure(figsize=(10,6))
			plt.plot(sample_steps, similarity_score_list, color='orange')
			plt.plot(sample_steps, test_similarity_score_list, color='goldenrod')
			plt.xlim(0)
			plt.ylim(0)
			plt.xlabel('step')
			plt.ylabel('SI')
			plt.savefig(f'{args.save_dir_samples}/{args.model_name}_SI.png')
			
			plt.figure(figsize=(10,6))
			plt.plot(total_loss_list, color='red')
			plt.plot(sample_steps, test_loss_list, color='firebrick')
			plt.xlim(0)
			plt.ylim(0)
			plt.xlabel('step')
			plt.ylabel('loss')
			plt.savefig(f'{args.save_dir_samples}/{args.model_name}_total_loss_curve.png')

			plt.figure(figsize=(10,6))
			plt.semilogy(total_loss_list, color='red')
			plt.semilogy(sample_steps, test_loss_list, color='firebrick')
			plt.xlim(0)
			plt.ylim(0)
			plt.xlabel('step')
			plt.ylabel('loss')
			plt.grid(True)
			plt.savefig(f'{args.save_dir_samples}/{args.model_name}_total_loss_curve_logscale.png')

			plt.figure(figsize=(10,6))
			plt.plot(running_loss_list, color='darkred')
			plt.xlim(0)
			plt.ylim(0)
			plt.xlabel('step')
			plt.ylabel('running mean loss')
			plt.savefig(f'{args.save_dir_samples}/{args.model_name}_running_mean_loss_curve.png')

			plt.figure(figsize=(10,6))
			plt.semilogy(running_loss_list, color='darkred')
			plt.xlim(0)
			plt.ylim(0)
			plt.xlabel('step')
			plt.ylabel('running mean loss')
			plt.grid(True)
			plt.savefig(f'{args.save_dir_samples}/{args.model_name}_running_mean_loss_curve_logscale.png')

			wandb.log(
				{
					"step": step,
					"loss": total_loss,
					"running_mean_loss": running_mean_loss,
				}
			)
			total_loss.backward()
			optimizer.step()

			pbar.set_description(f'loss: {total_loss.item():.4f}')

			step += 1
			pbar.update(1)
		
		data_dict['similarity_train']=similarity_score_list
		data_dict['similarity_test']=test_similarity_score_list
		data_dict['train_loss']=total_loss_list
		data_dict['test_loss']=test_loss_list
		data_dict['running_mean_loss']=running_loss_list

		np.savez(f'{args.save_dir_samples}/losses_and_similarity.npz', **data_dict)
		
		np.save(f'./samples/structures/while_training/{args.save_name}_generated_eps.npy', generated_eps)
		np.save(f'./samples/structures/while_training/{args.save_name}_original_eps.npy', original_eps)

if __name__ == '__main__':
	# usage: python3 train_controlnet_rectangle.py ./large_model/configs/args_test.yaml
	config_path = sys.argv[1]

	parser = argparse.ArgumentParser(description="Arguments for the controlnet model")
	with open(config_path, "r") as file:
		config = yaml.safe_load(file)

	# Update the parser's default values with the loaded configurations
	args = argparse.Namespace(**config)

	sweep_configuration = {
		"method": "grid",
		"metric": {
			"loss": {"value": "loss", "goal": "minimize"},
        	"similarity": {"value": "similarity", "goal": "maximize"}
		},
		"parameters": {
			"start_lr": {"values": args.start_lr},
			"end_lr": {"values": args.end_lr},
			"train_num_steps": {"values": args.train_num_steps}
		}
	}
	sweep_id = wandb.sweep(sweep=sweep_configuration, project=args.wandb_project)

	def sweep_function():
		with wandb.init() as run:
			args.start_lr=run.config.start_lr
			args.end_lr=run.config.end_lr
			args.train_num_steps=run.config.train_num_steps
			main(args)

	wandb.agent(sweep_id, function=sweep_function, count=4)

