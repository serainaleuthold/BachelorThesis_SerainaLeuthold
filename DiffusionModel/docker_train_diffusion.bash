#!/bin/bash
out_file='log_train_diffusion.txt'

docker run -v /home/chenkaim/scripts/models/ControlNet_EM:/workspace \
           -v /media/lts0/chenkaim/:/media/lts0/chenkaim/ \
           -v /media/ps2/chenkaim/:/media/ps2/chenkaim/ \
           --gpus all --rm --device=/dev/nvidia1 --device=/dev/nvidiactl --device=/dev/nvidia-uvm \
           -tid --ipc=host --name train_diffusion denoising_diffusion_pytorch:latest bash -c\
            "cd /workspace/train_diffusion
                python train.py
            
                > ${out_file} 2>&1"