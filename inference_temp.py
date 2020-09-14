#coding=utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision.transforms as transforms

import argparse
import os
import time
from networks import *
from spec_dataloader import SpectraDataset, SpectraDataLoader

import math
import numpy as np
import sys
import cv2
import pdb
import pickle
import ntpath
import sys
from arpys import dl

#torch.cuda.set_device(0)


def get_energy(energy_scale, pred):
    val = math.modf((len(energy_scale)-1)*(pred+1)/2)
    energy = energy_scale[int(val[1])] + val[0]*(energy_scale[int(val[1])+1]-energy_scale[int(val[1])]) 
    return energy

def get_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default = "gold_fermi_detection")
    parser.add_argument("--gpu_ids", default = "")
    parser.add_argument('-j', '--workers', type=int, default=1)
    parser.add_argument('-b', '--batch-size', type=int, default=4)
    parser.add_argument('--filename', default = None)
    parser.add_argument('--energy_axis', default = None)
    parser.add_argument('--momentum_axis', default = None)

    parser.add_argument("--fine_width", type=int, default = 768)
    parser.add_argument("--fine_height", type=int, default = 320)
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoint_store', help='save checkpoint infos')

    opt = parser.parse_args()
    return opt


def get_pred(energy_scale, pred):
    val = math.modf((len(energy_scale)-1)*(pred+1)/2)
    energy = energy_scale[int(val[1])] + val[0]*(energy_scale[int(val[1])+1]-energy_scale[int(val[1])])
    return energy


def get_energy_values(filename, output, opt, ind1, ind2):
    try:
        with open(os.path.join('./../../data/rishabh/gold', filename), 'rb') as f :
            D = pickle.load(f)
    except:
        with open(os.path.join('./../../data/rishabh/gold/new', filename), 'rb') as f :
            D = pickle.load(f)
    energy_scale, momentum_scale, data, ef_guess = D
    predicted_energy = get_pred(energy_scale[ind1:len(energy_scale)-ind2], output)
    return predicted_energy, ef_guess


def main():
    opt = get_opt()
    print(opt)

    if(opt.filename == None):
        print('No filename given to the programme')
        sys.exit()

    spec = np.load(opt.filename)
    spectra = spec['spectra'].astype('float64')
    spectra = transforms.ToTensor()(spectra)
    #scaling and normalization of spectra manually as there was some problems in transforms
    spectra = (spectra - torch.min(spectra))/(torch.max(spectra)-torch.min(spectra))
    spectra = ((spectra-0.5)/0.5).type('torch.DoubleTensor')
    model = Spec_unet()
    model.load_state_dict(torch.load('./checkpoint_store/gold_training_full/step_060000.pth'))
    model.eval()
    model.double()
    print(spec['filename'], " ", spec['ind1'], " ", spec['ind2'], " ", spec['ind3'], " ", spec['ind4']);
    with torch.no_grad():
        output = model(torch.unsqueeze(spectra, dim = 0))
        print(output)
        predicted_energy, actual_energy = get_energy_values(str(spec['filename']), output, opt, spec['ind1'], spec['ind2'])
        print(predicted_energy)
        print(actual_energy)


if __name__ == "__main__":
    main()

