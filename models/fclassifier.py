"""
Flat MC classifier code
By Thomas Mortier
"""

import torch
torch.manual_seed(2019)
import math
import sys
import time

import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import torch.optim as optim
import torchvision.models as models

from torch.autograd import Variable
from sklearn.metrics import accuracy_score

"""
Function which calculates accuracy of predictions
"""
def calculate_accuracy(outputs, labels):
    # transform labels and outputs
    if isinstance(outputs,torch.Tensor):
        labels_n = labels.cpu().data.numpy()
        outputs_n = outputs.cpu().data.numpy()
    else:
        labels_n = labels
        outputs_n = outputs

    # calculate accuracy
    acc = accuracy_score(np.argmax(labels_n, axis=1), np.argmax(outputs_n, axis=1))

    return acc

def get_num_learnable_params(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    return("NUMBER OF LEARNABLE PARAMS: " + str(params))

"""
Class which represents a flat multiclass classifier (FNet), together
with the UBOP algorithm, as implemented in predict.
"""
class FNet(nn.Module):
    def __init__(self, in_features, num_classes, dtype):
        super(FNet, self).__init__()
        # store size of incoming feature vector
        self.in_features = in_features
        # dtype of tensors
        self.dtype = dtype 
        # store number of classes
        self.m = num_classes
        # classifier
        self.classifier = nn.Sequential(
            #nn.Dropout(p=0.25),
            nn.Linear(in_features=self.in_features, out_features=self.m),
            nn.Softmax(dim=1)
        )
        
    """
    Forward pass function of FNet
    """
    def forward(self, x):
        return self.classifier(x)
        
    def EU(self, y, py, loss, params):
        if loss == "eloss":
            return (1-params[0]*(((len(y)-1)/(self.m-1))**params[1])) * py
        elif loss == "precision":
            return (1/len(y)) * py
        elif loss == "recall":
            return 1 * py
        elif loss == "fbeta":
            return ((1+(params[0]**2))/((params[0]**2)+len(y))) * py
        elif loss == "zaffalon":
            return (((params[0])/(len(y)))-((params[1])/(len(y)**2))) * py
        else:
            return "NOT IMPLEMENTED YET!"
    
    def checkstop(self, y, loss, params):
        if self.dtype == torch.cuda.FloatTensor:
            y_l = list(y.cpu().numpy())
        else:
            y_l = list(y.numpy())
        l = self.EU(y_l,1,loss,params)/(self.EU(y_l,1,loss,params)-self.EU(y_l+[0],1,loss,params))
        r = self.EU(y_l+[0,0],1,loss,params)/(self.EU(y_l+[0],1,loss,params)-self.EU(y_l+[0,0],1,loss,params))
        return l >= r
    
    """
    UBOP
    note that x is a batch of instances (pytorch currently only supports batch-wise prediction)
    """
    def predict(self, x, loss, params, early_stop=False):
        pred_list = []
        EU_list = []
        for x_i in range(x.shape[0]):
            inp = torch.cat([x[x_i,:].view(1,-1)]*x.shape[0],dim=0)
            # sort probabilities in decreasing order
            p, ind = torch.sort(self(inp)[0,:],descending=True)
            ind += 1
            y = []
            y_best = []
            EU_best = 0
            for i in range(1,len(p)+1):
                y = ind[:i]
                py = torch.sum(p[:i])
                EU = self.EU(y,py,loss,params)
                if EU_best < EU:
                    EU_best = EU
                    y_best = y
                elif early_stop and self.checkstop(y,loss,params):
                    break
            if self.dtype == torch.cuda.FloatTensor:
                pred_list.append(list(y_best.cpu().numpy()))
            else:
                pred_list.append(list(y_best.numpy()))
            EU_list.append(EU_best.item())
        return pred_list, EU_list

"""
Flat softmax neural network 
"""
class FMCC(nn.Module):
    def __init__(self,m,gpu,patience,vgg,ds,trdl=None,vldl=None):
        super(FMCC, self).__init__()
        # register number of classes, dataloader object, etc.
        self.nclass = m
        self.trdl = trdl
        self.vldl = vldl
        self.gpu = gpu
        if gpu:
            self.dtype = torch.cuda.FloatTensor
        else:
            self.dtype = torch.FloatTensor
            
        self.patience = patience
        self.vgg = vgg
        self.dataset = ds

        # feature extraction part
        if vgg:
            self.features = nn.Sequential(
                *list(models.vgg16_bn(pretrained=True).features.children())
            )
            self.ft_size = 25088
        elif "PROTEIN" in ds:
            self.embedding = nn.Conv1d(26,50,1)
            self.ft_cnn = nn.Sequential(
                nn.Conv1d(50,64,3),
                nn.BatchNorm1d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                nn.ReLU(),
                nn.MaxPool1d(2),
                nn.Conv1d(64,128,4),
                nn.BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                nn.ReLU(),
                nn.MaxPool1d(2),
                nn.Conv1d(128,256,5),
                nn.BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                nn.ReLU(),
                nn.MaxPool1d(2),
                nn.Conv1d(256,512,6),
                nn.BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                nn.ReLU(),
                nn.MaxPool1d(2),
            )
            if gpu:
                self.embedding.cuda()
                self.ft_cnn.cuda()
            out_embedding = self.embedding(torch.randn(32,26,1000).type(self.dtype))
            out_cnn = self.ft_cnn(out_embedding)
            self.ft_size_cnn = out_cnn.shape
            print("CNN SHAPE = {0}".format(self.ft_size_cnn))
            self.ft_size = self.ft_size_cnn[1]*self.ft_size_cnn[2]
            self.fund = nn.Sequential(
                nn.Linear(17929,1024),
                nn.BatchNorm1d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                nn.ReLU(),
                nn.Linear(1024,128),
                nn.BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                nn.ReLU()
            )
            self.final = nn.Sequential(
                nn.Linear(128+self.ft_size,5012),
                nn.BatchNorm1d(5012, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                nn.ReLU(),
                nn.Linear(5012,1024),
                nn.BatchNorm1d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                nn.ReLU(),
                nn.Linear(1024,128),
                nn.BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                nn.ReLU()
            )
            self.ft_size = 128
        else:
            self.features = nn.Sequential(
                nn.Conv2d(3, 16, kernel_size=(4, 4)),
                nn.BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, padding=0,ceil_mode=False),
                nn.Conv2d(16, 32, kernel_size=(3, 3)),
                nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2,ceil_mode=False),
                nn.Conv2d(32, 64, kernel_size=(3, 3)),
                nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2,ceil_mode=False),
                nn.Conv2d(64, 128, kernel_size=(3, 3)),
                nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2,ceil_mode=False),
                nn.Conv2d(128, 256, kernel_size=(3, 3)),
                nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2,ceil_mode=False)
            )
            self.ft_size = 6400

        # classification part
        self.fnet = FNet(self.ft_size, self.nclass, self.dtype)

    def forward(self,x):
        if "PROTEIN" in self.dataset:
            x, fundx = x[0],x[1]
            x = self.embedding(x)
            x = self.ft_cnn(x)
            if not x.is_contiguous():
                x = x.contiguous()
            x = x.view(-1,self.ft_size_cnn[1]*self.ft_size_cnn[2])
            fundx = self.fund(fundx)
            x = torch.cat([x, fundx],1)
            x = self.final(x)
            x = self.fnet(x)
            return x
        else:
            x = self.features(x)
            x = x.view(-1,self.ft_size)
            x = self.fnet(x)
            return x

    def train_model(self,ne,lr,ft=False,verbose=True):
        # fix weights of convnet
        if "PROTEIN" not in self.dataset:
            for p in self.features.parameters():
                p.requires_grad = ft
    
        # print model
        if verbose:
            print(self)
        print(get_num_learnable_params(self))
        
        # loss function
        criterion = nn.BCELoss()

        if not ft:
            if "PROTEIN" in self.dataset:
                optimizer = optim.Adam(self.parameters(), lr=lr)
            else:
                optimizer = optim.Adam(self.fnet.parameters(), lr=lr)
        else:
            optimizer = optim.Adam(self.parameters(), lr=lr)

        # intialize variables for patience mechanics
        min_val_loss = sys.maxsize
        patience_cnt = 0
        for epoch in range(ne):  # loop over the dataset multiple times
            train_running_loss = 0.0
            train_part_running_loss = 0.0
            train_time = 0.0
            val_running_loss = 0.0
            val_acc = 0.0

            # loop over training data
            counter_train = 0
            for i, data in enumerate(self.trdl, 0):
                if "PROTEIN" in self.dataset:
                    inputs, inputs_fund, labels = data
                    inputs, inputs_fund, labels = Variable(inputs.type(self.dtype)), Variable(inputs_fund.type(self.dtype)), Variable(labels.type(self.dtype), requires_grad=False)
                else:
                    inputs, labels = data
                    inputs, labels = Variable(inputs.type(self.dtype)), Variable(labels.type(self.dtype), requires_grad=False)

                # set model to training mode
                self.train()

                # zero the parameter gradients
                optimizer.zero_grad()

                start_time = time.time()
                if "PROTEIN" in self.dataset:
                    outputs = self([inputs, inputs_fund])
                else:
                    outputs = self(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                train_time += time.time()-start_time

                # print statistics
                train_part_running_loss += loss.item()
                train_running_loss += loss.item()
                counter_train += 1

            # loop over validation data
            counter_val = 0
            for i, data in enumerate(self.vldl, 0):
                # set model to evaluation mode and
                self.eval()

                if "PROTEIN" in self.dataset:
                    inputs, inputs_fund, labels = data
                    inputs, inputs_fund, labels = Variable(inputs.type(self.dtype)), Variable(inputs_fund.type(self.dtype)), Variable(labels.type(self.dtype))
                    outputs = self([inputs, inputs_fund]) 
                else:
                    inputs, labels = data
                    inputs, labels = Variable(inputs.type(self.dtype)), Variable(labels.type(self.dtype))   
                    outputs = self(inputs) 
                                   
                loss = criterion(outputs, labels)

                # print statistics
                val_running_loss += loss.item()
                if "PROTEIN" in self.dataset:
                    val_acc += calculate_accuracy(self([inputs, inputs_fund]), labels)
                else:
                    val_acc += calculate_accuracy(self(inputs), labels)
                counter_val += 1
            print("EPOCH {0}: lossTr={1:.3f}   timeTr={2:.4f}   lossVal={3:.3f}   accVal={4:.3f}".format(epoch+1,train_running_loss/counter_train,train_time/counter_train,val_running_loss/counter_val,val_acc/counter_val))
            self.save_state("inter_models/{0}/FLAT_MCC_{1}_{2}".format(self.dataset,lr,epoch+1))

            # check if early stopping applies
            valLoss = round(val_running_loss/counter_val,4)
            if valLoss < min_val_loss:
                min_val_loss = valLoss
                patience_cnt = 0
            else:
                patience_cnt += 1
                if patience_cnt == self.patience:
                    print("[info] early stopping applied after exceeding patience counter of " + str(self.patience))
                    break
        print('FINISHED TRAINING\n\n')

    def save_state(self,filename,verbose=False):
        if verbose:
            print("SAVING STATE OF MODEL...")
        torch.save(self.state_dict(), filename+".pt")
        if verbose:
            print("DONE!\n\n")

    def load_state(self,filename):
        print("loading state model...")
        self.load_state_dict(torch.load(filename))












