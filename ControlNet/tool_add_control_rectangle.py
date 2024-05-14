# usage:
# python tool_add_control_rectangle.py configs/args_test.yaml /media/ps2/leuthold/denoising-diffusion-pytorch/results_freeform/model-freeform_first-best.pt ./models/control_freeform_ini.ckpt

import sys
import os
import argparse
import yaml

import torch
from learners_controlled_rectangle import CombinedModel

def get_node_name(name, parent_name):
    if len(name) <= len(parent_name):
        return False, ''
    p = name[:len(parent_name)]
    if p != parent_name:
        return False, ''
    return True, name[len(parent_name):]

def main(args, input_path, output_path):
    model = CombinedModel(args)

    pretrained_weights = torch.load(input_path, map_location="cpu")
    if 'state_dict' in pretrained_weights:
        pretrained_weights = pretrained_weights['state_dict']  
    if 'model' in pretrained_weights:
        pretrained_weights=pretrained_weights['model']

    scratch_dict = model.state_dict()

    target_dict = {}
    for k in scratch_dict.keys():
        is_control, name = get_node_name(k, 'control_')
        if is_control:
            copy_k = 'model.diffusion_' + name
        else:
            copy_k = k
        if copy_k in pretrained_weights:
            target_dict[k] = pretrained_weights[copy_k].clone()
        else:
            target_dict[k] = scratch_dict[k].clone()
            print(f'These weights are newly added: {k}')

    model.load_state_dict(target_dict, strict=True)
    torch.save(model.state_dict(), output_path)
    print('Done.')

if __name__ == '__main__':
    config_path = sys.argv[1]
    input_path = sys.argv[2]
    output_path = sys.argv[3]

    assert os.path.exists(config_path), "configuration file doesn't exist."
    assert os.path.exists(input_path), 'Input model does not exist.'
    assert not os.path.exists(output_path), 'Output filename already exists.'
    assert os.path.exists(os.path.dirname(output_path)), 'Output path is not valid.'

    parser = argparse.ArgumentParser(description="Arguments for the controlnet model")
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    # Update the parser's default values with the loaded configurations
    args = argparse.Namespace(**config)

    main(args, input_path, output_path)

# python3 tool_add_control_rectangle.py './large_model/datasize/1K/configs/args_test.yaml' '../Diffusion/results/100K_Chenkai/model-simple_diff_10k-best.pt' './large_model/datasize/10K/models/model_10K.pt'
# python3 tool_add_control_rectangle.py './large_model/configs/args_test.yaml' '../Diffusion/results/100K_Chenkai/model-simple_diff_10k-best.pt' './large_model/models/last_model_thesis.pt'
# python3 tool_add_control_rectangle.py '/media/ps2/leuthold/denoising-diffusion-pytorch2/Control_Net/simplification2/configs/args_test.yaml' '/media/ps2/leuthold/denoising-diffusion-pytorch2/Diffusion/results/model-waveynet_skipped_large-best.pt' '/media/ps2/leuthold/denoising-diffusion-pytorch2/Control_Net/simplification2/models/control_waveynet_thesis'