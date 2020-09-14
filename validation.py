#coding=utf-8
# only segmentation prediction in the model
import torch
import torch.nn as nn
import torch.nn.functional as F

import argparse
import os
import time
from networks import *
from spec_dataloader import SpectraDataset, SpectraDataLoader

import numpy as np
import sys
import pdb

#torch.cuda.set_device(0)
def get_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default = "fermi_detection")
    parser.add_argument("--gpu_ids", default = "")
    parser.add_argument('-j', '--workers', type=int, default=1)
    parser.add_argument('-b', '--batch-size', type=int, default=4)
    
    parser.add_argument("--dataroot", default = "gold_batch_data")
    parser.add_argument("--datamode", default = "validation")
    parser.add_argument("--data_list", default = "validation.txt")
    parser.add_argument("--fine_width", type=int, default = 768)
    parser.add_argument("--fine_height", type=int, default = 320)
    parser.add_argument('--lr', type=float, default=0.0001, help='initial learning rate for adam')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoint_store', help='save checkpoint infos')
    parser.add_argument("--display_count", type=int, default = 20)
    parser.add_argument("--save_count", type=int, default = 500)
    parser.add_argument("--keep_step", type=int, default = 30000)
    parser.add_argument("--decay_step", type=int, default = 10000)
    parser.add_argument("--shuffle", action='store_true', help='shuffle input data')

    opt = parser.parse_args()
    return opt


def valid(opt, val_loader, model):
    #model.cuda()
    model.double()
    model.eval()
    li = []
    li_pred = []
    li_filename = []
    li_ind1 = []
    li_ind2 = []
    i = 0
    for step, inputs in enumerate(val_loader.data_loader):
        iter_start_time = time.time()
        spectra = inputs['spectra']
        fermi_energy = inputs['fermi_energy']
        fermi_energy = torch.unsqueeze(fermi_energy, 1)
            
        output = model(spectra)
        temp = torch.squeeze(output-fermi_energy, 1)
        li += (list(temp.detach().numpy()))
        li_pred += (list(output.detach().numpy()))
        li_filename += inputs['filename']
        li_ind1 += (list(inputs['ind1'].detach().numpy()))
        li_ind2 += (list(inputs['ind2'].detach().numpy()))
        i+=4
        print(i)

    for i in range(len(li)):
        li[i] = np.abs(li[i])
    pdb.set_trace()

def main():
    opt = get_opt()
    print(opt)
   
    # create dataset 
    val_dataset = SpectraDataset(opt)

    # create dataloader
    val_loader = SpectraDataLoader(opt, val_dataset)
    val = Spec_unet()

    model = Spec_unet()
    model.load_state_dict(torch.load('./checkpoint_store/gold_training/step_020000.pth'))
    
    valid(opt, val_loader, model)
    

if __name__ == "__main__":
    main()

