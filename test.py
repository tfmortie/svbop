import argparse
import sys
import linecache
import torch
import ast
import time



import numpy as np
import data as d
import pandas as pd
import torch.utils.data as Data
import torchvision.transforms as transforms
import models.fclassifier as fc
import models.hclassifier as hc

from torch.utils.data.sampler import SubsetRandomSampler
from multiprocessing import set_start_method
from torch.autograd import Variable
from itertools import compress

def stringify(paramlist):
    ret_str = ""
    for p in paramlist:
        ret_str = ret_str + str(p) + "_"

    return(ret_str[:-1])

def num_obs_csv(file):
    number_lines = sum(1 for line in open(file))
    return(number_lines-1)

def main_flat(pathmodel,pathcsv,bs,pat,vgg,gpu,loss,params,store):
    print("CREATE DATALOADER FOR {0}...".format(pathcsv))
    # string conversions
    params = ast.literal_eval(params)
    # define transformations
    transformations = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
    dataset = d.CustomDataset(pathcsv, transformations)
    m = dataset.get_m()

    # obtain test loader
    test_loader = Data.DataLoader(dataset,
                                batch_size=bs,
                                shuffle=False,
                                num_workers=4,
                                pin_memory=gpu
                                )
    print("DONE!\n\n")
    print("LOAD FLAT MODEL FOR {0}...".format(pathcsv))
    
    # create model
    net = fc.FMCC(dataset.get_m(),gpu,pat,vgg,pathcsv.split("/")[-2])
    if gpu:
        net.cuda()
    net.load_state(pathmodel)
    net.eval()
    print("DONE!\n\n")
    print("OBTAIN PREDICTIONS ON TEST SET...")
    # create empty containers which will contain the posterior's, targets, predictions
    posteriors = []
    targets = []
    predictions = []
    runtime = []
    # start processing test data
    for i, data in enumerate(test_loader, 0):
        # transform data to valid pytorch datatypes
        if gpu:
            dtype = torch.cuda.FloatTensor
        else:
            dtype = torch.FloatTensor

        inputs, labels = data
        inputs, labels = Variable(inputs.type(dtype)), Variable(labels.type(dtype))
        targs = labels.cpu().data.numpy()
        # store predictions and targets
        posteriors.extend(net(inputs).cpu().data.numpy().tolist())
        targets.extend(targs.tolist())
        start = time.time()
        predictions.extend(net.fnet.predict(net.features(inputs).view(-1,net.ft_size),loss,params)[0])
        end = time.time()
        runtime.append(end-start)
        
    print("[info] total time to calculate predictions: {0}".format(np.sum(np.asarray(runtime))))
    
    # create dataframe
    dfdata = {"posterior": posteriors, "target": targets, "prediction": predictions}
    df = pd.DataFrame(data=dfdata)

    # store information if necessary
    if store:
        print("SAVING POSTERIORS, TARGETS, PREDICTIONS TO out/...")
        df.to_csv("out/flat_{0}_{1}_{2}.csv".format(pathcsv.split("/")[-2],loss,params))
        
    print("[info] shape...")
    print(df.shape)
    print("DONE!\n\n")
    
    return df

def main_hierarchical(pathmodel,struct,pathcsv,bs,pat,vgg,gpu,loss,params,epsilon,store):
    print("CREATE DATALOADER FOR " + pathcsv + "...")
    # string conversions
    struct = ast.literal_eval(struct)
    params = ast.literal_eval(params)
    epsilon = float(epsilon)
    # define transformations
    transformations = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
    dataset = d.CustomDataset(pathcsv, transformations)
    m = dataset.get_m()

    # obtain test loader
    test_loader = Data.DataLoader(dataset,
                                batch_size=bs,
                                shuffle=False,
                                num_workers=4,
                                pin_memory=gpu
                                )
    print("DONE!\n\n")
    print("LOAD HIERARCHICAL MODEL FOR {0}...".format(pathcsv))
    # create model
    net = hc.HMCC(struct.copy(),gpu,pat,vgg,pathcsv.split("/")[-2])
    if gpu:
        net.cuda()
    net.load_state(pathmodel)
    net.eval()
    print("DONE!\n\n")
    print("OBTAIN PREDICTIONS ON TEST SET...")
    
    
    # create empty containers which will contain the posterior's, targets, predictions
    posteriors = []
    targets = []
    predictions = []
    runtime = []
    # start processing test data
    for i, data in enumerate(test_loader, 0):
        # transform data to valid pytorch datatypes
        if gpu:
            dtype = torch.cuda.FloatTensor
        else:
            dtype = torch.FloatTensor

        inputs, labels = data
        inputs, labels = Variable(inputs.type(dtype)), Variable(labels.type(dtype))
        targs = labels.cpu().data.numpy()
        # store predictions and targets
        posteriors.extend(net(inputs).cpu().data.numpy().tolist())
        targets.extend(targs.tolist())
        start = time.time()
        predictions.extend(net.hnet.predict(net.features(inputs).view(-1,net.ft_size),loss,params,epsilon)[0])
        end = time.time()
        runtime.append(end-start)
        
    print("[info] total time to calculate predictions: {0}".format(np.sum(np.asarray(runtime))))
    
    # create dataframe
    dfdata = {"posterior": posteriors, "target": targets, "prediction": predictions}
    df = pd.DataFrame(data=dfdata)

    # store information if necessary
    if store:
        print("SAVING POSTERIORS, TARGETS, PREDICTIONS TO out/...")
        df.to_csv("out/hierarchical_{0}_{1}_{2}_{3}.csv".format(pathcsv.split("/")[-2],loss,params,epsilon))
        
    print("[info] shape...")
    print(df.shape)
    print("DONE!\n\n")
    
    return df

def main(args):
    parser = argparse.ArgumentParser(description="CAHMCC TESTING")
    parser.add_argument("-pm", "--pathmodel", type=str, default="/", required=True)
    parser.add_argument("-pcsv","--pathcsv",type=str,default="/",required=True)
    parser.add_argument("-s","--struct",type=str,default="[]",required=False)
    parser.add_argument("-b","--batchsize",type=int,default=32,required=False)
    parser.add_argument("-pa","--patience",type=int,default=5,required=False)
    parser.add_argument("-m","--model",type=str,default="flat",choices=["flat", "hierarchical"],required=False)
    parser.add_argument("--vgg", dest="vgg", action='store_true')
    parser.add_argument("--no-vgg", dest="vgg", action='store_false')
    parser.add_argument("--gpu", dest="gpu", action='store_true')
    parser.add_argument("--no-gpu", dest="gpu", action='store_false')
    parser.add_argument("-l", "--loss", type=str, default="eloss",required=False)
    parser.add_argument("-p","--params", type=str, default="[1.0, 1.0]", required=False)
    parser.add_argument("-epsilon", "--epsilon", type=float, default=0.0, required=False)
    parser.add_argument("--store", dest="store", action='store_true')
    parser.add_argument("--not-store", dest="store", action='store_false')

    parser.set_defaults(vgg=True)
    parser.set_defaults(gpu=True)
    parser.set_defaults(store=True)

    args = parser.parse_args(args)

    # intitiate call based on type of model
    if args.model == "flat":
        main_flat(args.pathmodel,
                      args.pathcsv,
                      args.batchsize,
                      args.patience,
                      args.vgg,
                      args.gpu,
                      args.loss,
                      args.params,
                      args.store)
    else:
        main_hierarchical(args.pathmodel,
                      args.struct,
                      args.pathcsv,
                      args.batchsize,
                      args.patience,
                      args.vgg,
                      args.gpu,
                      args.loss,
                      args.params,
                      args.epsilon,
                      args.store)

# top level code
if __name__ == '__main__':
    # pass arguments to main function
    main(sys.argv[1:])