import os
import csv
import random
import numpy as np
import torch.utils.data as data
import torchvision.transforms as transforms
import nibabel as nib
import torch

class Dataset(data.Dataset):
    'characterizes a dataset for pytorch'
    def __init__(self, dir_data, list_IDs, patch_size):
        self.dir_data = dir_data
        self.list_IDs = list_IDs
        self.ps       = patch_size

    def __len__(self):
        'denotes the total number of samples'
        return len(self.list_IDs)

    def __getitem__(self, item):
        'Generates one sample of data'
        if len(self.list_IDs[item]) == 2:
            path_mr = os.path.join(self.dir_data,self.list_IDs[item][1])
            I = nib.load(path_mr).get_fdata(dtype=np.float32)
            I = I[20:170,20:210,10:160]
            I = np.nan_to_num(I);
            I = (I-np.min(I))/(np.max(I)-np.min(I))*256
            I = torch.from_numpy(I)
            patches = I.unfold(2, self.ps, self.ps).unfold(1, self.ps, self.ps).unfold(0, self.ps, self.ps)
            patches = patches.contiguous().view(-1, self.ps, self.ps, self.ps)
        label = self.list_IDs[item][0]
        return patches, label



class Utils:
    def __init__(self, path, demographics, data):
        self.path_demographics = os.path.join(path,demographics)
        self.path_data         = os.path.join(path,data)
        self.path              = path

    def get_group(self):
        labels = []
        with open(self.path_demographics) as f:
            spamreader = csv.reader(f, delimiter=',', quotechar='|')
            for row in spamreader:
                l = int(int(row[17])*1.0)
                labels.append(l)
        return labels

    def create_dataset(self):
        labels = self.get_group();
        i=0;
        list_ids = []
        with open(self.path_data) as f:
            spamreader = csv.reader(f, delimiter=',', quotechar='|')
            for row in spamreader:
                pdata = '{}.nii.gz'.format(row[2])
                list_ids.append((int(labels[i]), pdata))
                i = i+1
        return Dataset(self.path, list_ids, 64);
