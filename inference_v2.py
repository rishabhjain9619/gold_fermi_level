#coding=utf-8
import random
import torch

import torchvision.transforms as transforms

import argparse
import os
from networks import *

import math
import numpy as np
import cv2
import pickle
import ntpath

#torch.cuda.set_device(0)


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


def main():
    opt = get_opt()
    print(opt)

    if(opt.filename == None):
        print('No filename given to the programme')
        sys.exit()

    with open(opt.filename, 'rb') as f :
            D = pickle.load(f)
    energy_scale, momentum_scale, data, ef_guess = D
    final_pred = []
    model = Spec_unet()
    model.load_state_dict(torch.load('./checkpoint_store/gold_training_temp/step_060000_v2.pth'))
    model.eval()
    model.double()
    #averaging over various augmented data spectra for better accuracy similar to what was done in training
    for i in range(10):
        ind3 = random.randint(0, int(0.25*momentum_scale.shape[0]))
        ind4 = random.randint(0, int(0.25*momentum_scale.shape[0]))
        data_temp = data[ind3:len(momentum_scale)-ind4, :]
        #converting the data to a fixed size
        spectra = cv2.resize(data_temp, dsize=(opt.fine_width, opt.fine_height), interpolation=cv2.INTER_CUBIC).astype('float64')
        spectra = transforms.ToTensor()(spectra)
        #scaling and normalizing the spectra
        spectra = (spectra - torch.min(spectra))/(torch.max(spectra)-torch.min(spectra))
        spectra = ((spectra-0.5)/0.5).type('torch.DoubleTensor')
        
        with torch.no_grad():
            pred = model(torch.unsqueeze(spectra, dim = 0))
            energy = get_pred(energy_scale, pred)
            print("predicted energy at augmentation step", i, "is:-",  energy)
            final_pred.append(energy)

    print()
    print("Final predicted energy is:- " , sum(final_pred)/len(final_pred))
    print("Actual energy", ef_guess)


if __name__ == "__main__":
    main()

