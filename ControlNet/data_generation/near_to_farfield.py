# -*- coding: utf-8 -*-
import numpy as np
from PIL import Image
import random

from stratton_chu_ceviche_rectangle import strattonChu2D

import time
import sys,os

from tqdm import tqdm
import argparse

import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
font = {'size'   : 16}

matplotlib.rc('font', **font)

def plot_helper(i, eps, Hz, Ex, Ey, src, output_dir):
    fig, ax = plt.subplots(1,5)
    
    im = ax[0].imshow(eps)
    divider = make_axes_locatable(ax[0])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)
    ax[0].set_title("dielectric")

    im = ax[1].imshow(src.real)
    divider = make_axes_locatable(ax[1])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)
    ax[1].set_title("source")

    im = ax[2].imshow(Hz.real)
    divider = make_axes_locatable(ax[2])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)
    ax[2].set_title("Hz")

    im = ax[3].imshow(Ex.real)
    divider = make_axes_locatable(ax[3])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)
    ax[3].set_title("Ex")

    im = ax[4].imshow(Ey.real)
    divider = make_axes_locatable(ax[4])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)
    ax[4].set_title("Ey")

    plt.savefig(output_dir + f"/debug_{i}.png", dpi=500)
    plt.close()

def main(args):
    eps = np.load(args.input_dir+"/"+f"input_eps_{args.name}_{args.data_type}.npy")
    Hz = np.load(args.input_dir+"/"+f"Hz_out_forward_RI_{args.name}_{args.data_type}.npy")
    Ex = np.load(args.input_dir+"/"+f"Ex_out_forward_RI_{args.name}_{args.data_type}.npy")
    Ey = np.load(args.input_dir+"/"+f"Ey_out_forward_RI_{args.name}_{args.data_type}.npy")

    # using existing images as input:
    if args.debug:
        num_devices = 10
    else:
        num_devices = eps.shape[0]

    spacing = int((args.Nx-2*args.pml_x-args.device_length)/3)

    nx, ny = args.device_length, args.device_length
    print("nx. ny: ", nx, ny)

    xc = (-args.Nx/2 + args.pml_x+2*spacing + int(nx/2))*args.dL*1e-9
    # xc = 0
    yc = 0

    Rx = int(nx/2 + 1*spacing/2)*args.dL*1e-9
    Ry = int(ny/2 + 1*spacing/2)*args.dL*1e-9

    sx = args.Nx*args.dL*1e-9
    sy = args.Ny*args.dL*1e-9

    Far_Ex = np.empty([num_devices, args.N_theta], dtype = complex)
    Far_Ey = np.empty([num_devices, args.N_theta], dtype = complex)
    Far_Hz = np.empty([num_devices, args.N_theta], dtype = complex)

    tic = time.time()
    for i in tqdm(range(num_devices)):
        shift_devices_debug=0
        i=i+shift_devices_debug
        this_Ex = Ex[i][:,:,0] + 1j*Ex[i][:,:,1]
        this_Ey = Ey[i][:,:,0] + 1j*Ey[i][:,:,1]
        this_Hz = Hz[i][:,:,0] + 1j*Hz[i][:,:,1]
        i=i-shift_devices_debug
        theta_obs, Far_Ex[i], Far_Ey[i], Far_Hz[i] = strattonChu2D(i, args.dL*1e-9, sx, sy, args.Nx, args.Ny, xc, yc, Rx, Ry, args.WAVELENGTH*1e-9, this_Ex.T, this_Ey.T, -this_Hz.T, N_theta=args.N_theta, debug=args.debug)

        if i<10: #args.debug:
            plt.figure(figsize=(8,18))

            i=i+shift_devices_debug

            tmp = eps[i]/np.max(eps[i])
            i=i-shift_devices_debug
            tmp[args.pml_x+2*spacing+int(nx/2) + int(nx/2 + 1*spacing/2), int(args.Ny/2)-int(ny/2 + 1*spacing/2):int(args.Ny/2)+int(ny/2 + 1*spacing/2)] += 1
            tmp[args.pml_x+2*spacing+int(nx/2) - int(nx/2 + 1*spacing/2), int(args.Ny/2)-int(ny/2 + 1*spacing/2):int(args.Ny/2)+int(ny/2 + 1*spacing/2)] += 1
            tmp[args.pml_x+2*spacing+int(nx/2) - int(nx/2 + 1*spacing/2) : args.pml_x+2*spacing+int(nx/2) + int(nx/2 + 1*spacing/2), int(args.Ny/2)-int(ny/2 + 1*spacing/2)] += 1
            tmp[args.pml_x+2*spacing+int(nx/2) - int(nx/2 + 1*spacing/2) : args.pml_x+2*spacing+int(nx/2) + int(nx/2 + 1*spacing/2), int(args.Ny/2)+int(ny/2 + 1*spacing/2)] += 1
            plt.subplot(4, 2, 1)
            plt.imshow(tmp.T)
            plt.title("N2F box")

            i=i+shift_devices_debug
            plt.subplot(4, 2, 3)
            plt.imshow(Ex[i][:,:,0].T)
            i=i-shift_devices_debug
            plt.title("Ex_near_field")
            plt.subplot(4, 2, 4, projection='polar')
            plt.plot(theta_obs, np.abs(Far_Ex[i][::-1])**2, label='Ex')
            plt.legend()
            plt.subplot(4, 2, 5)
            i=i+shift_devices_debug
            plt.imshow(Ey[i][:,:,0].T, label='Ey')
            i=i-shift_devices_debug
            plt.title("Ey_near_field")
            plt.subplot(4, 2, 6, projection='polar')
            plt.plot(theta_obs, np.abs(Far_Ey[i][::-1])**2, c='k', label='Ey')
            plt.legend()
            plt.subplot(4, 2, 7)
            i=i+shift_devices_debug
            plt.imshow(Hz[i][:,:,0].T, label='Hz')
            i=i-shift_devices_debug
            plt.title("Hz_near_field")
            plt.subplot(4, 2, 8, projection='polar')
            plt.plot(theta_obs, np.abs(Far_Hz[i][::-1])**2, '-.', c='r', label='Hz')
            plt.legend()

            plt.savefig(args.output_dir+"/"+f"n2f_{args.name}_{args.data_type}_{i}.png", dpi=200)
            plt.close()

    toc = time.time()
    print(f"Device finished: {num_devices}, The total time of the data generation is {toc - tic}s")

    np.save(args.output_dir+"/"+f"Far_Hz_{args.name}_{args.data_type}.npy", Far_Hz)
    np.save(args.output_dir+"/"+f"Far_Ex_{args.name}_{args.data_type}.npy", Far_Ex)
    np.save(args.output_dir+"/"+f"Far_Ey_{args.name}_{args.data_type}.npy", Far_Ey)

if __name__ == '__main__':

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--input_dir', type=str, help='location to load npy files', default="/media/ps2/leuthold/denoising-diffusion-pytorch/EM_fields/waveynet")
    argparser.add_argument('--output_dir', type=str, help='location to save farfield', default="/media/ps2/leuthold/denoising-diffusion-pytorch/EM_fields/waveynet")
    argparser.add_argument('--name', type=str, help='name extension for files (e.g. model name used)', default="corrected_multiple")
    argparser.add_argument('--data_type', type=str, help='original or generated data', default="original")
    
    argparser.add_argument('--WAVELENGTH', type=float, help='wavelength in nm', default=800)
    argparser.add_argument('--dL', type=float, help='wavelength in nm', default=16)
    argparser.add_argument('--Nx', type=int, help='number of pixels in x', default=750)
    argparser.add_argument('--Ny', type=int, help='number of pixels in y', default=750)
    argparser.add_argument('--n_mat', type=float, help='material index', default=2.0)
    argparser.add_argument('--n_sub', type=float, help='substrate index', default=1.0)
    argparser.add_argument('--pml_x', type=int, help='number of pixels in x for PML layer', default=40)
    argparser.add_argument('--pml_y', type=int, help='number of pixels in y for PML layer', default=40)
    argparser.add_argument('--device_length', type=int, help='number of pixels for the side length of the device', default=250)

    argparser.add_argument('--N_theta', type=int, help='number of angles to record far field', default=361)
    argparser.add_argument('--debug', type=int, help='if set to 1, debug data generation with visualization', default=0)

    args = argparser.parse_args()

    if not os.path.isdir(args.output_dir):
        raise ValueError(f"directory not found: {args.output_dir}")

    main(args)

    # python3 near_to_farfield.py --input_dir "/media/ps2/leuthold/denoising-diffusion-pytorch/Control_Net/samples/near_fields" --output_dir "/media/ps2/leuthold/denoising-diffusion-pytorch/Control_Net/samples/far_fields"
    # python3 near_to_farfield.py --debug 1
