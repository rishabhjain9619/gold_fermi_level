import torch
import torch.nn as nn
from torch.nn import init
import numpy as np
import torch.nn.functional as F
import os
from torchsummaryX import summary

def save_checkpoint(model, save_path):
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))

    torch.save(model.cpu().state_dict(), save_path)

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('Linear') != -1:
        init.normal(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def weights_init_xavier(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.xavier_normal_(m.weight.data, gain=0.02)
    elif classname.find('Linear') != -1:
        init.xavier_normal_(m.weight.data, gain=0.02)
    elif classname.find('BatchNorm2d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('BatchNorm2d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def init_weights(net, init_type='normal'):
    print('initialization method [%s]' % init_type)
    if init_type == 'normal':
        net.apply(weights_init_normal)
    elif init_type == 'xavier':
        net.apply(weights_init_xavier)
    elif init_type == 'kaiming':
        net.apply(weights_init_kaiming)
    else:
        raise NotImplementedError('initialization method [%s] is not implemented' % init_type)


class Spec_unet(nn.Module):
    def __init__(self, input_nc = 1, nf = 8, fine_width = 320, fine_height = 768, max_nf = 64):
        super(Spec_unet, self).__init__()
        norm_layer = nn.BatchNorm2d
        use_bias = False
        conv_downsample = [
            nn.Conv2d(input_nc, nf, kernel_size=7, padding=3, bias=use_bias),
            norm_layer(nf),
            nn.LeakyReLU(0.1)]
        nc = nf
        nf*=2
        for i in range(6):
            conv_downsample += [
                nn.Conv2d(nc, min(max_nf, nf), kernel_size=3, stride=2, padding=1, bias=use_bias),
                nn.Dropout(0.2),
                norm_layer(nf),
                nn.LeakyReLU(0.1)]
            nc = min(max_nf, nf)
            nf = min(nf*2, max_nf)

        conv_downsample += [
            nn.Conv2d(nf, 8, kernel_size=3, padding=1, bias=use_bias),
            norm_layer(8),
            nn.LeakyReLU(0.1)]


        self.conv_downsample = nn.Sequential(*conv_downsample)
        
        self.linear = nn.Sequential(
                nn.Linear(8*12*5, 32),
                nn.Dropout(0.2),
                nn.BatchNorm1d(num_features=32),
                nn.LeakyReLU(0.1),
                nn.Linear(32, 1),
                nn.Tanh()
                )
        


    def forward(self, input):
        downsample = self.conv_downsample(input)
        downsample = downsample.view(-1, 12*5*8)
        output = self.linear(downsample)
        return output


class Spec_unet_smooth(nn.Module):
    def __init__(self, input_nc = 1, nf = 8, fine_width = 320, fine_height = 768, max_nf = 128):
        super(Spec_unet_smooth, self).__init__()
        norm_layer = nn.BatchNorm2d
        use_bias = False
        conv_downsample = [
            nn.Conv2d(input_nc, nf, kernel_size=7, padding=3, bias=use_bias),
            norm_layer(nf),
            nn.LeakyReLU(0.1)]
        nc = nf
        nf*=2
        for i in range(6):
            conv_downsample += [
                nn.Conv2d(nc, min(max_nf, nf), kernel_size=3, stride=[1, 2], padding=1, bias=use_bias),
                nn.Dropout(0.2),
                norm_layer(nf),
                nn.LeakyReLU(0.1)]
            nc = min(max_nf, nf)
            nf = min(nf*2, max_nf)

        conv_downsample += [
            nn.Conv2d(nf, 8, kernel_size=3, padding=1, bias=use_bias),
            norm_layer(8),
            nn.LeakyReLU(0.1)]


        self.conv_downsample = nn.Sequential(*conv_downsample)
        
        self.linear = nn.Sequential(
                nn.Linear(8*12*320, 320*4),
                nn.Dropout(0.2),
                nn.BatchNorm1d(num_features=320*4),
                nn.LeakyReLU(0.1),
                nn.Linear(320*4, 320),
                nn.Tanh()
                )
        


    def forward(self, input):
        downsample = self.conv_downsample(input)
        downsample = downsample.view(-1, 12*320*8)
        output = self.linear(downsample)
        return output

if __name__ == '__main__':
    model = Spec_unet_smooth()
    init_weights(model, 'kaiming')
    print(model)
    arch = summary(model, torch.rand(2,1,320,768))
    print(arch)
