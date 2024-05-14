import torch
from torch.utils.data import DataLoader

import sys
sys.path.append("../rectangle")
sys.path.append("../dataloaders")
from learners_controlled_rectangle_large import CombinedModel
from train_util import *
import argparse
import yaml
import itertools
from tqdm import tqdm
import os
import math
import numpy as np

import ceviche

def angle_distance(ang1, ang2):
    diff = ang1-ang2
    return min(abs(diff), abs(diff+360), abs(diff-360))

def inverse_farfield(angle, phase_deg, magnitude=1.5, half_spread=20, N_theta=360):
    farfield_real = torch.zeros(N_theta)
    farfield_imag = torch.zeros(N_theta)

    for i in range(N_theta):
        farfield_real[i] = 0 if angle_distance(i, angle) > half_spread else magnitude*np.cos(phase_deg*np.pi/180)*(np.cos((i-angle)/half_spread*np.pi)+1)
        farfield_imag[i] = 0 if angle_distance(i, angle) > half_spread else magnitude*np.sin(phase_deg*np.pi/180)*(np.cos((i-angle)/half_spread*np.pi)+1)

    return torch.stack((farfield_real, farfield_imag), dim=-1)[None]

def sim_near_field(args, eps_r, source_x_start, spacing):
    wavelength = args.wl
    omega = 2 * np.pi * C_0 / wavelength
    k0 = 2 * np.pi / wavelength

    F = fdfd_hz(omega, args.dL*1e-9, eps_r, [args.pml_x, args.pml_y])
    
    source_amp = 64e9/args.dL**2
    source = np.zeros(eps_r.shape, dtype=complex)

    source[source_x_start, args.pml_y+spacing:-args.pml_y-spacing] = source_amp
    source[source_x_start-1, args.pml_y+spacing:-args.pml_y-spacing] = -source_amp*np.exp(-1j*k0*args.dL*1e-9)

    Ex_forward, Ey_forward, Hz_forward = F.solve(source)
    
    return eps_r, Hz_forward, Ex_forward, Ey_forward, source

def get_superpixel(model, args, angle, phase_deg, magnitude = 1.5, FDFD_verify=False):
    step = 0
    with torch.inference_mode():
        farfield = inverse_farfield(angle, phase_deg, magnitude=magnitude).cuda()
        print("dimension of farfield: ", farfield.shape)
        generated_samples = model.sample(farfield, batch_size = args.num_parallel_samples)

    if FDFD_verify:
        sim_score = analyze_inverse_design_source_in(0, args, farfield[0].cpu().numpy(), generated_samples.cpu().numpy(), magnitude=magnitude, name=f"angle_{angle}_phase_{phase_deg}")
        print("sim score: mean", np.mean(sim_score), "max: ", np.max(sim_score), "min: ", np.min(sim_score))
        return generated_samples[np.argmax(sim_score)]
        
    else:
        return generated_samples[0]

def pad_airs(design, air_pixels):
    assert air_pixels%2 == 0
    return np.pad(design, ((0,0),(air_pixels//2, air_pixels//2)))

def main(args, fl, half_size, air_gap=0.2):
    ############## model loading #############
    save_dir_samples = args.model_saving_path + '/samples/'
    if not os.path.exists(save_dir_samples):
        os.makedirs(save_dir_samples)
    # Configs
    resume_path = args.model_saving_path + '/models/' + f'{args.model_name}.pt'
    model = CombinedModel(args)

    # load the state_dict from a new CombinedModel with trained diffusion weights and empty controlnet weights
    # which is produced by tool_add_control.py
    model.load_state_dict(torch.load(resume_path)['state_dict'])
    model = model.cuda()
    print(f"model weights loaded from {resume_path}")

    # First use CPU to load models. PyTorch Lightning will automatically move it to GPUs.
    model.learning_rate = args.start_lr
    model.diffusion_locked = args.diffusion_locked
    model.only_mid_control = args.only_mid_control

    ############## inverse design ############ 
    spacing = 100
    Nx = 2*args.pml_x + 2*spacing + args.image_sizex + round(1.2*fl*1000/args.dL)
    Ny = 2*args.pml_y + 2*spacing + round(2*half_size*1000/args.dL)
    print(f"total sim size: {Nx}, {Ny}")
    x_start = args.pml_x + spacing
    source_x_start = args.pml_x + spacing - 4

    total_eps = np.ones([Nx, Ny])

    half_size_pixel = round(half_size*1000/args.dL)
    fl_pixel = round(fl*1000/args.dL)
    air_gap_pixel = round(air_gap*1000/args.dL)

    N_designs = int(half_size_pixel/(args.image_sizey+air_gap_pixel))
    for i in tqdm(range(N_designs)):
        y_pixel = (i+1/2) * (args.image_sizey + air_gap_pixel)
        angle_deg = -np.arctan(y_pixel/fl_pixel)*180/np.pi
        phase_deg = (y_pixel**2+fl_pixel**2)**.5*args.dL*1e-9/args.wl*360

        design = get_superpixel(model, args, angle_deg, phase_deg, magnitude=1.3, FDFD_verify=False) # shape: (1, 32, 96)
        design = pad_airs(design[0].cpu().numpy(), air_gap_pixel)*(args.n_mat-1) + 1
        total_eps[x_start:x_start+args.image_sizex, Ny//2+i*(args.image_sizey+air_gap_pixel):Ny//2+(i+1)*(args.image_sizey+air_gap_pixel)] = design
        total_eps[x_start:x_start+args.image_sizex, Ny//2-(i+1)*(args.image_sizey+air_gap_pixel):Ny//2-i*(args.image_sizey+air_gap_pixel)] = design[:,::-1]


    eps_r, Hz_forward, Ex_forward, Ey_forward, source = sim_near_field(args, total_eps, source_x_start, spacing)

    plt.figure(figsize=(5,10))
    plt.subplot(2,1,1)
    plt.imshow(eps_r)
    plt.colorbar()
    plt.subplot(2,1,2)
    intensity = Ex_forward.real*Ex_forward.real+Ex_forward.imag*Ex_forward.imag+Ey_forward.real*Ey_forward.real+Ey_forward.imag*Ey_forward.imag
    print("intensity: ", intensity)
    plt.imshow(10*np.log10(intensity/np.max(intensity)))
    plt.colorbar()
    plt.clim((-20,0))
    plt.savefig(f"./inverse_design/lens/samples/lens_fl{fl}_size_{2*half_size}_dB20.png", transparent=True, dpi=500, bbox_inches="tight", pad_inches=0)

    plt.clim((-10,0))
    plt.savefig(f"./inverse_design/lens/samples/lens_fl{fl}_size_{2*half_size}_dB10.png", transparent=True, dpi=500, bbox_inches="tight", pad_inches=0)

    # plt.subplot(4,1,3)
    # plt.imshow(Ex_forward.real)
    # plt.colorbar()
    # plt.subplot(4,1,4)
    # plt.imshow(Ey_forward.real)
    # plt.colorbar()
    

if __name__ == '__main__':
    # usage: python3 inverse_design_lens.py /workspace/Control_Net/large_model/configs/args_test_interpolation.yaml 23 40
    config_path = sys.argv[1]
    fl = float(sys.argv[2]) # in um
    half_size = float(sys.argv[3]) # in um

    parser = argparse.ArgumentParser(description="Arguments for the controlnet model")
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    # Update the parser's default values with the loaded configurations
    args = argparse.Namespace(**config)
    main(args, fl, half_size)
