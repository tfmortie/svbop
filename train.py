"""
Train code for image data
"""

import argparse
import sys
import linecache
import torch
torch.manual_seed(2019)
import ast
import csv

import numpy as np
import data as d
import torch.utils.data as Data
import torchvision.transforms as transforms
import models.fclassifier as fc
import models.hclassifier as hc
import models.tree_generator as tg

from torch.utils.data.sampler import SubsetRandomSampler
from multiprocessing import set_start_method

def stringify(paramlist):
    ret_str = ""
    for p in paramlist:
        ret_str = ret_str + str(p) + "_"

    return(ret_str[:-1])

def main_flat(pathcsv,valsize,shuffle,ne,lr,bs,pat,ft,vgg,gpu,random_seed):
    # configuration for GPU
    try:
        set_start_method('spawn')
    except RuntimeError:
        pass

    print("CREATE DATALOADER FOR {0} ...".format(pathcsv))
    # define transformations
    transformations = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
    if "PROTEIN" in pathcsv:
        dataset = d.ECDataset(pathcsv)
    else:
        dataset = d.ImageDataset(pathcsv, transformations)

    # initialize samples for training and validation set
    num_samples = len(dataset)
    num_classes = dataset.get_m()
    print("[info] shape train/val data: ({0},{1})".format(num_samples,num_classes))
    indices = list(range(num_samples))
    split = int(np.floor(valsize * num_samples))

    if shuffle:
        np.random.seed(random_seed)
        np.random.shuffle(indices)

    train_idx, valid_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    # obtain training and validation loaders
    train_loader = Data.DataLoader(dataset,
                                batch_size=bs,
                                sampler=train_sampler,
                                num_workers=4,
                                pin_memory=gpu
                                )

    val_loader = Data.DataLoader(dataset,
                                batch_size=bs,
                                sampler=valid_sampler,
                                num_workers=4,
                                pin_memory=gpu
                                )
    print("DONE!\n\n")
    print("CREATE MODEL...")
    # create model
    net = fc.FMCC(num_classes,gpu,pat,vgg,pathcsv.split("/")[-2],train_loader,val_loader)
    if gpu:
        net.cuda()
    print("DONE!\n\n")
    print("START TRAINING FLAT MCC MODEL ON {0}...".format(pathcsv))
    net.train_model(ne,lr,ft=ft)
    print("DONE!\n\n")
    # save trained model
    net.save_state("FLAT_MCC_TRAIN_{0}_{1}".format(stringify([valsize,shuffle,ne,lr,bs,pat,ft,vgg]),pathcsv.split("/")[-2]), verbose=True)

def main_hierarchical(pathcsv,struct,learns,k,valsize,shuffle,ne,lr,bs,pat,ft,vgg,gpu,random_seed):
    # configuration for GPU
    try:
        set_start_method('spawn')
    except RuntimeError:
        pass
    # check whether the hierarchy needs to be learned or not, with k number of hierarchies
    if not learns:
        k=1
        
    for k_i in range(k):
        print("CREATE DATALOADER FOR {0}...".format(pathcsv))
        # define transformations
        transformations = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
        
        # create dataset object
        if "PROTEIN" in pathcsv:
            dataset = d.ECDataset(pathcsv)
        else:
            dataset = d.ImageDataset(pathcsv, transformations)

        # initialize samples for training and validation set
        num_samples = len(dataset)
        num_classes = dataset.get_m()
        print("[info] shape train/val data: ({0},{1})".format(num_samples,num_classes))
        
        if not learns:
            struct = ast.literal_eval(struct)
        else:
            # create random hierarchy
            treegen = tg.TreeGenerator(num_classes)
            struct = treegen.GenerateHierarchy(m_s=1)
            # and save to wd
            struct_text = str(struct).replace(' ','')
            with open("/".join(pathcsv.split("/")[0:-1])+"/"+"hierarchy_random.txt", "w") as h_file:
                print(f"{struct_text}", file=h_file)
                
        num_nodes = len(struct)
        indices = list(range(num_samples))
        split = int(np.floor(valsize * num_samples))

        if shuffle:
            np.random.seed(random_seed)
            np.random.shuffle(indices)

        train_idx, valid_idx = indices[split:], indices[:split]
        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)

        # obtain training and validation loaders
        train_loader = Data.DataLoader(dataset,
                                batch_size=bs,
                                sampler=train_sampler,
                                num_workers=4,
                                pin_memory=gpu
                                )

        val_loader = Data.DataLoader(dataset,
                                batch_size=bs,
                                sampler=valid_sampler,
                                num_workers=4,
                                pin_memory=gpu
                                )
        if len(struct) <= 500:
            print("Hierarchy: {0}".format(struct))
        print("DONE!\n\n")
        print("CREATE MODEL...")
        # create model
        net = hc.HMCC(struct,gpu,pat,vgg,pathcsv.split("/")[-2],train_loader,val_loader)
        if gpu:
            net.cuda()
        print("DONE!\n\n")
        print("START TRAINING HIERARCHICAL MCC MODEL ON {0}...".format(pathcsv))
        net.train_model(ne,lr,ft=ft)
        print("DONE!\n\n")
        # save trained model
        net.save_state("HIERARCHICAL_MCC_TRAIN_{0}_{1}".format(stringify([valsize,shuffle,ne,lr,bs,pat,ft,vgg,num_nodes,learns,k_i]),pathcsv.split("/")[-2]),verbose=True)

def main(args):
    parser = argparse.ArgumentParser(description="CAHMCC TRAINING")
    parser.add_argument("-pcsv","--pathcsv",type=str,default="/",required=True)
    parser.add_argument("-s","--struct",type=str,default="[]",required=False)
    parser.add_argument("--learns", dest="learns", action='store_true')
    parser.add_argument("--no-learns", dest="learns", action='store_false')
    parser.add_argument("-k","--k",type=int,default=1)
    parser.add_argument("-v","--valsize",type=float,default=0.2,required=False)
    parser.add_argument("--shuffle", dest="shuffle", action='store_true')
    parser.add_argument("--no-shuffle", dest="shuffle", action='store_false')
    parser.add_argument("-e","--epochs",type=int,default=100,required=False)
    parser.add_argument("-l","--learnrate",type=float,default=0.001,required=False)
    parser.add_argument("-b","--batchsize",type=int,default=32,required=False)
    parser.add_argument("-pa","--patience",type=int,default=5,required=False)
    parser.add_argument("--ft", dest="ft", action='store_true')
    parser.add_argument("--no-ft", dest="ft", action='store_false')
    parser.add_argument("-m","--model",type=str,default="flat",choices=["flat", "hierarchical"],required=False)
    parser.add_argument("--vgg", dest="vgg", action='store_true')
    parser.add_argument("--no-vgg", dest="vgg", action='store_false')
    parser.add_argument("--gpu", dest="gpu", action='store_true')
    parser.add_argument("--no-gpu", dest="gpu", action='store_false')
    parser.add_argument("-rs","--randomseed",type=int, default=2018, required=False)

    parser.set_defaults(learns=False)
    parser.set_defaults(shuffle=True)
    parser.set_defaults(ft=False)
    parser.set_defaults(vgg=True)
    parser.set_defaults(gpu=True)

    args = parser.parse_args(args)

    # intitiate call based on type of model
    if args.model == "flat":
        main_flat(args.pathcsv,
                      args.valsize,
                      args.shuffle,
                      args.epochs,
                      args.learnrate,
                      args.batchsize,
                      args.patience,
                      args.ft,
                      args.vgg,
                      args.gpu,
                      args.randomseed)
    else:
        main_hierarchical(args.pathcsv,
                      args.struct,
                      args.learns,
                      args.k,
                      args.valsize,
                      args.shuffle,
                      args.epochs,
                      args.learnrate,
                      args.batchsize,
                      args.patience,
                      args.ft,
                      args.vgg,
                      args.gpu,
                      args.randomseed)

# top level code
if __name__ == '__main__':
    # pass arguments to main function
    main(sys.argv[1:])
