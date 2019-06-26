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

def main_flat(pathmodel,struct,pathcsv,bs,pat,vgg,gpu,loss,params,store):
    print("CREATE DATALOADER FOR {0}...".format(pathcsv))
    # we need the struct for BU-RBOP
    with open(struct, 'r') as reader:
        # read struct file
        struct = reader.read()
    struct = ast.literal_eval(struct)
    # sort in increasing order of size of sets
    struct = list(np.array(struct)[list(np.argsort(np.array([len(x) for x in struct])))])
    # transform to index list
    struct = [list(np.array(x)-1) for x in struct]
    params = ast.literal_eval(params)
    # define transformations
    transformations = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
    if "PROTEIN" in pathcsv:
        dataset = d.ECDataset(pathcsv)
    else:
        dataset = d.ImageDataset(pathcsv, transformations)
        
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
    predictions_sc = []
    predictions_nsc = []
    predictions_burbop = []
    nv_sc = []
    nv_nsc = []
    nv_burbop = []
    runtime_sc = []
    runtime_nsc = []
    runtime_burbop = []
    # start processing test data
    for i, data in enumerate(test_loader, 0):
        # transform data to valid pytorch datatypes
        if gpu:
            dtype = torch.cuda.FloatTensor
        else:
            dtype = torch.FloatTensor
            
        if "PROTEIN" in pathcsv:
            inputs, inputs_fund, labels = data
            inputs, inputs_fund, labels = Variable(inputs.type(net.dtype)), Variable(inputs_fund.type(net.dtype)), Variable(labels.type(net.dtype))
        else:
            inputs, labels = data
            inputs, labels = Variable(inputs.type(net.dtype)), Variable(labels.type(net.dtype))           
        
        targs = labels.cpu().data.numpy()
        # store predictions and targets
        if "PROTEIN" in pathcsv:
            posteriors.extend(net([inputs, inputs_fund]).cpu().data.numpy().tolist())
        else:   
            posteriors.extend(net(inputs).cpu().data.numpy().tolist())
            
        targets.extend(targs.tolist())
        if "PROTEIN" in pathcsv:
            inputs = net.embedding(inputs)
            inputs = net.ft_cnn(inputs)
            if not inputs.is_contiguous():
                inputs = inputs.contiguous()
            inputs = inputs.view(-1,net.ft_size_cnn[1]*net.ft_size_cnn[2])
            fundx = net.fund(inputs_fund)
            inputs = torch.cat([inputs, fundx],1)
            inputs = net.final(inputs)
        else:
            inputs = net.features(inputs).view(-1,net.ft_size)
        start = time.time()
        predl_sc, _ , nvl_sc = net.fnet.predict(inputs,loss,params,True)
        end = time.time()
        predictions_sc.extend(predl_sc)
        runtime_sc.extend([(end-start)/len(targs.tolist())]*len(targs.tolist()))
        nv_sc.extend(nvl_sc)
        start = time.time()
        predl_nsc, _ , nvl_nsc = net.fnet.predict(inputs,loss,params)
        end = time.time()
        predictions_nsc.extend(predl_nsc)
        runtime_nsc.extend([(end-start)/len(targs.tolist())]*len(targs.tolist()))
        nv_nsc.extend(nvl_nsc)
        start = time.time()
        predl_burbop, _ , nvl_burbop = net.fnet.predict_burbop(inputs,loss,params,struct)
        end = time.time()
        predictions_burbop.extend(predl_burbop)
        runtime_burbop.extend([(end-start)/len(targs.tolist())]*len(targs.tolist()))
        nv_burbop.extend(nvl_burbop)
        
    print("[info] total time to calculate predictions: {0}".format(np.sum(np.asarray(runtime_sc))+np.sum(np.asarray(runtime_nsc))+np.sum(np.asarray(runtime_burbop))))
    
    # create dataframe    
    dfdata = {"pred_ubop_sc": predictions_sc,"pred_ubop_nsc": predictions_nsc, "pred_burbop": predictions_burbop, "posterior": posteriors, "target": targets, "runtime_sc": runtime_sc, "nv_sc": nv_sc, "runtime_nsc": runtime_nsc, "nv_nsc": nv_nsc, "runtime_burbop": runtime_burbop, "nv_burbop": nv_burbop}
    df = pd.DataFrame(data=dfdata)

    # store information if necessary
    if store:
        print("SAVING TO out/...")
        df.to_csv("out/all_csv_1605/flat_{0}_{1}_{2}.csv".format(pathcsv.split("/")[-2],loss,params),index=False)
        
    print("[info] shape...")
    print(df.shape)
    print("DONE!\n\n")
    
    return df

def main_hierarchical(pathmodel,struct,pathcsv,bs,pat,vgg,gpu,loss,params,epsilon,store):
    print("CREATE DATALOADER FOR " + pathcsv + "...")
    # string conversions
    with open(struct, 'r') as reader:
        # read struct file
        struct = reader.read()
    struct = ast.literal_eval(struct)
    params = ast.literal_eval(params)
    epsilon = float(epsilon)
    # define transformations
    transformations = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
    # create dataset object
    if "PROTEIN" in pathcsv:
        dataset = d.ECDataset(pathcsv)
    else:
        dataset = d.ImageDataset(pathcsv, transformations)
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
    
    # create empty containers which will contain the targets, predictions, runtime
    posteriors = []
    targets = []
    predictions_rbop = []
    predictions_hsubop = []
    nv_rbop = []
    nv_hsubop = []
    runtime_rbop = []
    runtime_hsubop = []
    # start processing test data
    for i, data in enumerate(test_loader, 0):
        # transform data to valid pytorch datatypes
        if gpu:
            dtype = torch.cuda.FloatTensor
        else:
            dtype = torch.FloatTensor

        if "PROTEIN" in pathcsv:
            inputs, inputs_fund, labels = data
            inputs, inputs_fund, labels = Variable(inputs.type(net.dtype)), Variable(inputs_fund.type(net.dtype)), Variable(labels.type(net.dtype))
        else:
            inputs, labels = data
            inputs, labels = Variable(inputs.type(net.dtype)), Variable(labels.type(net.dtype))   
        targs = labels.cpu().data.numpy()
        # store predictions and targets
        targets.extend(targs.tolist())
        if "PROTEIN" in pathcsv:
            inputs = net.embedding(inputs)
            inputs = net.ft_cnn(inputs)
            if not inputs.is_contiguous():
                inputs = inputs.contiguous()
            inputs = inputs.view(-1,net.ft_size_cnn[1]*net.ft_size_cnn[2])
            fundx = net.fund(inputs_fund)
            inputs = torch.cat([inputs, fundx],1)
            inputs = net.final(inputs)
        else:
            inputs = net.features(inputs).view(-1,net.ft_size)
        posteriors.extend(net.hnet.predict(inputs).cpu().data.numpy().tolist())    
        start = time.time()
        predl_rbop, _, nvl_rbop = net.hnet.predict_rbop(inputs,loss,params,epsilon)
        end = time.time()
        predictions_rbop.extend(predl_rbop)
        runtime_rbop.extend([(end-start)/len(targs.tolist())]*len(targs.tolist()))
        nv_rbop.extend(nvl_rbop)
        start = time.time()
        predl_hsubop, _, nvl_hsubop = net.hnet.predict_hsubop(inputs,loss,params,early_stop=True)
        end = time.time()
        predictions_hsubop.extend(predl_hsubop)
        runtime_hsubop.extend([(end-start)/len(targs.tolist())]*len(targs.tolist()))
        nv_hsubop.extend(nvl_hsubop)
        
    print("[info] total time to calculate predictions: {0}".format(np.sum(np.asarray(runtime_rbop))+np.sum(np.asarray(runtime_hsubop))))
    # create dataframe
    dfdata = {"pred_rbop": predictions_rbop, "pred_hsubop": predictions_hsubop, "posterior": posteriors, "target": targets, "runtime_rbop": runtime_rbop, "nv_rbop": nv_rbop, "runtime_hsubop": runtime_hsubop, "nv_hsubop": nv_hsubop}
    df = pd.DataFrame(data=dfdata)
    
    # store information if necessary
    if store:
        print("SAVING TO out/...")
        df.to_csv("out/all_csv_1605/hierarchical_{0}_{1}_{2}_{3}_{4}.csv".format(pathmodel.split("_")[-3],pathcsv.split("/")[-2],loss,params,epsilon),index=False)
        
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
                      args.struct,
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
