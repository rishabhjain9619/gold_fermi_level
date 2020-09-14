import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import os.path as osp


class SpectraDataset(data.Dataset):
    """SpectraDataset for fermi level detection.
    """
    def __init__(self, opt):
        super(SpectraDataset, self).__init__()
        # base setting
        self.opt = opt
        self.root = opt.dataroot
        self.datamode = opt.datamode # train or test or self-defined
        self.data_list = opt.data_list
        self.fine_height = opt.fine_height
        self.fine_width = opt.fine_width
        self.data_path = osp.join(opt.dataroot, opt.datamode)
        self.convert = transforms.Compose([  \
                transforms.ToTensor()
                ])
        self.normalize = transforms.Compose([  \
                transforms.Normalize((0.5, ), (0.5, ))])


        
        # load data list
        spectra_names = []
        with open(opt.data_list, 'r') as f:
            for line in f.readlines():
                spectra_names.append(line.rstrip().replace('.npz', '_smooth_levels.npz'))

        self.spectra_names = spectra_names

    def name(self):
        return "SpectraDataset"

    def __getitem__(self, index):
        spec_name = self.spectra_names[index]
        # load spectra
        spec = np.load(osp.join(self.data_path, spec_name))
        spectra = spec['spectra'].astype('float64') 
        spectra = self.convert(spectra)
        #scaling and normalization of spectra manually as there was some problems in transforms
        spectra = (spectra - torch.min(spectra))/(torch.max(spectra)-torch.min(spectra))
        spectra = ((spectra-0.5)/0.5).type('torch.DoubleTensor')
        #spectra = self.normalize(spectra)
        result = {
            'filename': str(spec['filename']),
            'spectra':   spectra,     # contains one spectra
            'fermi_energy': torch.tensor(spec['smooth_fermi_levels']), #for ground truth
            'ind1': torch.tensor(spec['ind1']),
            'ind2': torch.tensor(spec['ind2']),
            'spec_name': str(spec_name)
            }

        return result

    def __len__(self):
        return len(self.spectra_names)

class SpectraDataLoader(object):
    def __init__(self, opt, dataset):
        super(SpectraDataLoader, self).__init__()

        if opt.shuffle :
            train_sampler = torch.utils.data.sampler.RandomSampler(dataset)
        else:
            train_sampler = None

        self.data_loader = torch.utils.data.DataLoader(
                dataset, batch_size=opt.batch_size, shuffle=(train_sampler is None),
                num_workers=opt.workers, pin_memory=True, sampler=train_sampler)
        self.dataset = dataset
        self.data_iter = self.data_loader.__iter__()
       
    def next_batch(self):
        try:
            batch = self.data_iter.__next__()
        except StopIteration:
            self.data_iter = self.data_loader.__iter__()
            batch = self.data_iter.__next__()

        return batch


if __name__ == "__main__":
    print("Checking that the dataloader is correct for spectra loading!")
    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataroot", default = "gold_batch_data")
    parser.add_argument("--datamode", default = "train")
    parser.add_argument("--data_list", default = "train.txt")
    parser.add_argument("--fine_width", type=int, default = 768)
    parser.add_argument("--fine_height", type=int, default = 320)
    parser.add_argument("--shuffle", action='store_true', help='shuffle input data')
    parser.add_argument('-b', '--batch-size', type=int, default=4)
    parser.add_argument('-j', '--workers', type=int, default=1)
    
    opt = parser.parse_args()
    dataset = SpectraDataset(opt)
    data_loader = SpectraDataLoader(opt, dataset)

    print('Size of the dataset: %05d, dataloader: %04d' \
            % (len(dataset), len(data_loader.data_loader)))
    first_item = dataset.__getitem__(0)
    '''    
    while(1):
        first_batch = data_loader.next_batch()
        if(torch.isnan(first_batch['fermi_energy']).any()):
            print(first_batch['spec_name'])
            print(first_batch['fermi_energy'])
    '''
    first_batch = data_loader.next_batch()
    print(first_batch)
