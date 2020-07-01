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
        #select sample
        if len(self.list_IDs[item]) == 2:
            path_mr = os.path.join(self.dir_data,self.list_IDs[item][1])
            I = nib.load(path_mr).get_fdata(dtype=np.float32)
            I = I[20:170,20:210,10:160]
            I = np.nan_to_num(I);
            I = (I-np.min(I))/(np.max(I)-np.min(I))*256
            I = torch.from_numpy(I)

            patches = I.unfold(2, self.ps, self.ps).unfold(1, self.ps, self.ps).unfold(0, self.ps, self.ps)
            patches = patches.contiguous().view(-1, self.ps, self.ps, self.ps)
        else:
            # Loading MRI
            path_mr = os.path.join(self.dir_data,self.list_IDs[item][1])
            I = nib.load(path_mr).get_fdata(dtype=np.float32)
            # Loading brain mask            
            path_mask = os.path.join(self.dir_data,self.list_IDs[item][2])
            M = nib.load(path_mr).get_fdata(dtype=np.float32)
            
            # Remove everything outside of the ROI
            I = I*(M>0);
            I = np.nan_to_num(I);
            I = (I-np.min(I))/(np.max(I)-np.min(I))*256

            # Compute number of patch and length of each segment
            idx = np.argwhere(M>0)
            Nseg = 500
            Lseg = int(np.floor(len(idx)/Nseg))
            
            first = True;
            for i in range(0,len(idx),Lseg):
                # X corners
                xmin = max(idx[i][0]-32,0)
                xmax = min(idx[i][0]+32,I.shape[0]-1)
                # Y corners
                ymin = max(idx[i][1]-32,0)
                ymax = min(idx[i][1]+32,I.shape[1]-1)
                # Z corners
                zmin = max(idx[i][2]-32,0)
                zmax = min(idx[i][2]+32,I.shape[2]-1)

                if (zmax-zmin)==64 and (ymax-ymin)==64 and (xmax-xmin)==64:
                    subI = I[xmin:xmax, ymin:ymax, zmin:zmax]
                    subI = torch.from_numpy(subI)
                    subI = torch.unsqueeze(subI,0)
                    if first:
                        first = False
                        patches = subI
                    else:
                        patches = torch.cat((patches, subI))
            # take 300 first patches in the middle 
            bound = int(patches.size(0)/2)
            #print(bound)
            patches = patches[bound-50:bound+50:4,:,:,:]
        #print(patches.size())
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


    def create_dataset(self, datatype, skullstrip):
        labels = self.get_group();
        i=0;
        list_ids = []
        with open(self.path_data) as f:
            spamreader = csv.reader(f, delimiter=',', quotechar='|')
            for row in spamreader:
                if datatype=='t1':
                    pdata = row[2]
                elif datatype=='t2':
                    pdata = row[3]
                elif datatype=='pbd':
                    pdata = '{}.nii.gz'.format(row[6])
                elif datatype=='pbg':
                    pdata = 'PBG/pbg_{}-{}.nii.gz'.format(row[0], row[1])
                if skullstrip:
                    list_ids.append((int(labels[i]),pdata, row[4]))
                else:
                    list_ids.append((int(labels[i]), pdata))
                i = i+1
        return Dataset(self.path, list_ids, 64);
