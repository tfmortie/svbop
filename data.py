import os
import torch
import ast

import pandas as pd
import numpy as np

from PIL import Image
from torch.utils.data import Dataset
from sklearn import preprocessing as pp

"""Dataset wrapping images and labels for an arbitrary dataset

    Arguments:
        A CSV file path
        PIL transforms
"""
class ImageDataset(Dataset):
    def __init__(self, csv_path, transform=None):
        # store transformation pipeline
        self.transform = transform
        # get csv file with information
        tmp_df = pd.read_csv(csv_path)
        # all images present?
        assert tmp_df['full_path'].apply(lambda x: os.path.isfile(x)).all(), \
            "Some images referenced in the CSV file were not found"

        # store X and Y dataframes
        self.X = tmp_df['full_path']
        self.y = tmp_df['marginal_labels'].apply(lambda x: np.fromstring(x[1:-1],dtype=np.float32,sep=","))
        self.m = len(self.y[0])

    def __getitem__(self, index):
        img = Image.open(self.X[index])
        img = img.convert('RGB')
        if self.transform is not None:
            img = self.transform(img)

        label = torch.from_numpy(self.y[index])

        return img, label

    def __len__(self):
        return len(self.X.index)

    def get_m(self):
        return self.m

"""EC number dataset

    Arguments:
        A CSV file path
        PIL transforms
"""
class ECDataset(Dataset):
    #mpl = 4911
    mpl = 1000 # truncated sequences
    m = 3485 
    #m = 236 # when level 4 has been omitted
    #m = 64 # when level 3 has been omitted
    def __init__(self, csv_path, transform=None):
        # transform currently not supported for EC number data...
        self.transform = transform
        # get csv file with information
        tmp_df = pd.read_csv(csv_path)

        # store X and Y dataframes
        oh_matrix_labels = np.eye(self.m).astype(np.float32)
        self.X_seq = tmp_df['enzyme']
        self.X_fund = tmp_df['fund']
        self.y = np.asarray(tmp_df.label.apply(lambda x: oh_matrix_labels[x-1,:]).values)
        
        # create dictionary for family -> index mapping
        filepath = "/".join(csv_path.split("/")[0:-1])+"/fams.txt"  
        self.fam_dict = {}
        with open(filepath) as fp:  
           line = fp.readline()
           fid = 0
           while line:
               self.fam_dict[line.strip()] = fid
               line = fp.readline()
               fid += 1

    def __getitem__(self, index):
        prot = self.X_seq[index]
        prot = self.pad_sequence(prot)
        prot = self.oh_sequence(prot)
        fund = self.X_fund[index]
        fund_v = [0]*len(self.fam_dict)
        for f in ast.literal_eval(fund):
            fund_v[self.fam_dict[f]] = 1
        fund = torch.from_numpy(np.asarray(fund_v)) 
        label = torch.from_numpy(self.y[index])
        
        return prot, fund, label

    def __len__(self):
        return len(self.X_seq.index)
                        
    def get_m(self):
        return self.m
                                  
    def pad_sequence(self,x,trunc=0):
        if len(x) > self.mpl:
            x = x[:self.mpl]
        ret_x = x
        len_add = self.mpl-len(x)
        while(len_add > 0):
            ret_x += "["
            len_add-=1
        return ret_x
                        
    def oh_sequence(self,x,index=False):
        ret = []
        if index:
            for p in x:
                ret.append(ord(p)-65)
            return np.asarray(ret)
        else:
            oh_matrix = np.vstack([np.eye(26).astype(np.float32),np.zeros((1,26))])
            for p in x:
                ret.append(oh_matrix[ord(p)-65,:].reshape(26,1))
            return np.hstack(ret)
        
        return np.hstack(ret)
