#coding=utf-8
# only segmentation prediction in the model
import torch
import torch.nn as nn
import torch.nn.functional as F

import argparse
import os
import time
from networks import *
from spec_dataloader_smooth import SpectraDataset, SpectraDataLoader

import numpy as np
import sys

#torch.cuda.set_device(0)
def get_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default = "gold_training_smooth")
    parser.add_argument("--gpu_ids", default = "")
    parser.add_argument('-j', '--workers', type=int, default=3)
    parser.add_argument('-b', '--batch-size', type=int, default=4)
    
    parser.add_argument("--dataroot", default = "gold_batch_data")
    parser.add_argument("--datamode", default = "train")
    parser.add_argument("--data_list", default = "train.txt")
    parser.add_argument("--fine_width", type=int, default = 768)
    parser.add_argument("--fine_height", type=int, default = 320)
    parser.add_argument('--lr', type=float, default=0.0001, help='initial learning rate for adam')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoint_store', help='save checkpoint infos')
    parser.add_argument("--display_count", type=int, default = 20)
    parser.add_argument("--save_count", type=int, default = 500)
    parser.add_argument("--keep_step", type=int, default = 50000)
    parser.add_argument("--decay_step", type=int, default = 40000)
    parser.add_argument("--shuffle", action='store_true', help='shuffle input data')

    opt = parser.parse_args()
    return opt


def train(opt, train_loader, model):
    #model.cuda()
    model.double()
    model.train()
    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, betas=(0.5, 0.999))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda = lambda step: 1.0 -
            max(0, step - opt.keep_step) / float(opt.decay_step + 1))

    criterionL1 = nn.L1Loss()

    for step in range(opt.keep_step + opt.decay_step):
        optimizer.zero_grad()
        iter_start_time = time.time()
        inputs = train_loader.next_batch()

        spectra = inputs['spectra']
        fermi_energy = inputs['fermi_energy']
        #fermi_energy = torch.unsqueeze(fermi_energy, 1)

        #print(fermi_energy, fermi_energy.shape)
            
        output = model(spectra)
        loss_fermi = criterionL1(output, fermi_energy)

        loss_fermi.backward()

        #print(step)
        optimizer.step()
        if (step+1) % opt.display_count == 0:
            t = time.time() - iter_start_time
            print('step: %8d, time: %.3f, loss_fermi: %.4f ' 
                    % (step+1, t, loss_fermi.item(),), flush=True)

        if (step+1) % opt.save_count == 0:
            print('saving info', os.path.join(opt.checkpoint_dir, opt.name))
            save_checkpoint(model, os.path.join(opt.checkpoint_dir, opt.name, 'step_%06d.pth' % (step+1)))


def main():
    opt = get_opt()
    print(opt)
    print("Start to train stage: named: %s!" % (opt.name))
   
    # create dataset 
    train_dataset = SpectraDataset(opt)

    # create dataloader
    train_loader = SpectraDataLoader(opt, train_dataset)
    model = Spec_unet_smooth()
    init_weights(model, 'kaiming')
    train(opt, train_loader, model)
    
    print('Finished training, named: %s!' % (opt.name))


if __name__ == "__main__":
    main()

