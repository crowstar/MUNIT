"""
Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
from __future__ import print_function
from utils import get_config, get_data_loader_folder
from trainer import UNIT_Trainer
import argparse
from torch.autograd import Variable
from data import ImageFolder
import torchvision.utils as vutils
try:
    from itertools import izip as zip
except ImportError: # will be 3.x series
    pass
import sys
import torch
import os
# imports for cycleGAN codebase
from util import html
from util.visualizer import save_images
from collections import OrderedDict

# input_folder should contain testA and testB folders
# output_folder does not need to include results/folder, just folder
parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='configs/edges2handbags_folder', help='Path to the config file.')
parser.add_argument('--input_folder', type=str, help="input image folder")
parser.add_argument('--output_folder', type=str, help="output image folder")
parser.add_argument('--checkpoint', type=str, help="checkpoint of autoencoders")
parser.add_argument('--seed', type=int, default=1, help="random seed")
parser.add_argument('--output_path', type=str, default='.', help="path for logs, checkpoints, and VGG model weight")
parser.add_argument('--stream', type=str, default='full', help='toggles generating a2b or b2a stream. [a2b|b2a]')

opts = parser.parse_args()

torch.manual_seed(opts.seed)
torch.cuda.manual_seed(opts.seed)

# Load experiment setting
config = get_config(opts.config)
input_dim = config['input_dim_a']

# Setup data directories
folder = os.path.join(opts.input_folder)

# Setup data loaders
names = ImageFolder(folder, transform=None, return_paths=True)

data_loader = get_data_loader_folder(folder, 1, False, new_size=config['new_size'], crop=False)

# Setup model
config['vgg_model_path'] = opts.output_path

trainer = UNIT_Trainer(config)

state_dict = torch.load(opts.checkpoint)

trainer.gen_a.load_state_dict(state_dict['a'])
trainer.gen_b.load_state_dict(state_dict['b'])
trainer.cuda()
trainer.eval()

# A -> B
encode_a2z = trainer.gen_a.encode # encode from A
decode_z2b = trainer.gen_b.decode # decode to B 

# B -> A
encode_b2z = trainer.gen_b.encode # encode from B
decode_z2a = trainer.gen_a.decode # decode to A

# create website
web_dir = os.path.join(opts.output_folder, 'test_latest')
webpage = html.HTML(web_dir, 'Experiment = %s, Phase = Test' % opts.output_folder)

# Start testing
for (real, name) in zip(data_loader, names):
    print(name[1])

    # content_b_A means we started with image A, and the latent has been calculated from b->z
    
    # real_A -> Z -> fake_B
    with torch.no_grad():
        if opts.stream == 'a2b': 
            real_A =  real.cuda()
            content_a_A, _ = encode_a2z(real_A)
            fake_B = decode_z2b(content_a_A)

        if opts.stream == 'b2a':
            # real_B -> Z -> fake_A
            real_B = real.cuda()
            content_b_B, _ = encode_b2z(real_B)
            fake_A = decode_z2a(content_b_B)

    basename = os.path.basename(name[1])

    
    # make dict of output images to be saved to site
    visuals = OrderedDict()
    if opts.stream == 'a2b':
        visuals['real_A'] = real_A.data
        visuals['fake_B'] = fake_B.data
    if opts.stream == 'b2a':
        visuals['real_B'] = real_B.data
        visuals['fake_A'] = fake_A.data
        
    img_path = os.path.join(opts.output_folder,basename)
    save_images(webpage, visuals, img_path, aspect_ratio=1, width=256)

webpage.save()

