"""
Hierarchical MC classifier code 
By Thomas Mortier
"""

import torch
torch.manual_seed(2019)
import math
import sys
import csv
import heapq
import time

import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import torch.optim as optim
import torchvision.models as models
import torch.nn.init as init

from torch.autograd import Variable
from copy import deepcopy
from sklearn.metrics import accuracy_score

"""
Priority queue needed for the RBOP inference algorithm
"""
class PriorityList():
    def __init__(self):
        self.list = []

    def push(self,prob,node):
        heapq.heappush(self.list,[1-prob,node])

    def pop(self):
        return heapq.heappop(self.list)

    def remove_all(self):
        self.list = []
        
    def size(self):
        return len(self.list)

    def is_empty(self):
        return len(self.list) == 0

    def __repr__(self):
        ret_str = ""
        for l in self.list:
            ret_str+="({0:.2f},{1}), ".format(1-l[0],l[1].y)
        return ret_str

"""
Function which calculates accuracy of predictions
"""
def calculate_accuracy(outputs, labels):
    # transform labels and outputs
    if isinstance(labels,torch.Tensor):
        labels_n = labels.cpu().data.numpy()
    else:
        labels_n = labels
        
    # calculate accuracy
    acc = accuracy_score(np.argmax(labels_n, axis=1)+1, outputs)

    return acc
    
def get_num_learnable_params(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    return("NUMBER OF LEARNABLE PARAMS: " + str(params))

"""
Class which represents a node in HNet
"""
class HNode(nn.Module):
    def __init__(self, y, in_features, do_node = 0.0):
        super(HNode, self).__init__()
        self.y = y  # denotes the subset/singleton of class(es)
        self.in_features = in_features  # dimensionality of feature vector
        self.chn = nn.ModuleList()
        self.classifier = None  # contains a node classifier (only for internal nodes)
        self.do_node = do_node # dropout probability to be applied in each node 
        
    """
    Function which (recursively) adds a child node
    """        
    def add_child_node(self, y):
        # check if leaf or internal node
        if len(self.chn) > 0:
            ind = [i for i, j in enumerate(self.chn) if set(y).issubset(set(j.y))]
            # check if y is a subset of one of the children
            if len(ind) > 0:
                self.chn[ind[0]].add_child_node(y)
            else:
                self.chn.append(HNode(y, self.in_features))
                tot_len_y_chn = sum([len(c.y) for c in self.chn])
                if tot_len_y_chn == len(self.y):
                    self.classifier = nn.Sequential(
                        nn.Dropout(p=self.do_node),
                        nn.Linear(self.in_features, len(self.chn)),
                        nn.Softmax(dim=1)
                    )
        else:
            self.chn.append(HNode(y, self.in_features))
    
    """
    Forward pass function of (internal) node classifier
    """
    def forward(self, x):
        return self.classifier(x)
    
    """
    Comparator needed in case of equal keys in queue (key=prob)
    """
    def __lt__(self, other):
        return len(self.y) < len(other.y)

"""
Class which represents a hierarchical multiclass classifier (HNet), together
with the RBOP algorithm, as implemented in predict.
"""
class HNet(nn.Module):
    def __init__(self, in_features, struct, dtype, do_node = 0.0):
        super(HNet, self).__init__()
        # store size of incoming feature vector and dropbout prob for each node
        self.in_features = in_features
        self.n_do = do_node
        # structure of hierarchy (BFS)
        self.struct = deepcopy(struct)
        # dtype of tensors
        self.dtype = dtype 
        # store number of classes
        self.m = len(self.struct[0])
        # store dropout for each node
        self.do_node = do_node
        # root classifier
        self.root = HNode(struct.pop(0), self.in_features, self.do_node)
        # recursively construct the hierarchical tree classifier
        while (len(struct) > 0):
            self.root.add_child_node(struct.pop(0))
             
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
    
    """
    stopping criterion
    """
    def checkstop(self, y, loss, params):
        l = self.EU(y,1,loss,params)/(self.EU(y,1,loss,params)-self.EU(y+[0],1,loss,params))
        r = self.EU(y+[0,0],1,loss,params)/(self.EU(y+[0],1,loss,params)-self.EU(y+[0,0],1,loss,params))
        return l >= r
        
    """
    forward pass of HNet
    """
    def forward(self, x, l=None, train=True):
        if train:
            batch_losses = list(map(self._train,x,l))
            return batch_losses
        else:
            return list(map(self._argmax,x))
        
    def _argmax(self, x):
        node = self.root
        # peform argmax starting from root downwards to the leafs
        while(len(node.chn) != 0):
            p_argmax =np.argmax(node(x.unsqueeze(0)).cpu().data.numpy())
            node = node.chn[p_argmax]
        # now obtain class
        return node.y[0]
    
    def _train(self, x, l):
        crit = nn.BCELoss()
        node = self.root
        path_loss = []
        while len(node.chn) != 0:
            n_ind = [i for i, j in enumerate(node.chn) if set([l]).issubset(set(j.y))][0]
            size = len(node.chn)     
            oh_sel = Variable(torch.eye(size)[n_ind,:].type(self.dtype), requires_grad=False)
            path_loss.append(crit(node(x.unsqueeze(0)), oh_sel.unsqueeze(0)))
            node = node.chn[n_ind]
        return torch.sum(torch.stack(path_loss))
    
    """
    HS-UBOP
    note that x is a batch of instances (pytorch currently only supports batch-wise prediction)
    """
    def predict_hsubop(self, x, loss, params, early_stop=False):
        pred_list = []
        EU_list = []
        for x_i in range(x.shape[0]):
            Q = PriorityList()
            Q.push(1.,self.root)
            L = []
            y_best, y = [],[]
            EU_best, py = 0, 0
            # first sort probabilities
            while not Q.is_empty():
                yhat_prob, yhat = Q.pop()
                yhat_prob = 1.-yhat_prob
                if len(yhat.chn) == 0:
                    # we are at a leaf node, hence, perform UBOP iteration
                    y.extend(yhat.y)
                    py += yhat_prob
                    EU = self.EU(y,py,loss,params)
                    if EU_best < EU:
                        EU_best = EU
                        y_best = y.copy()
                    elif early_stop and self.checkstop(y,loss,params):
                        break
                else:
                    yhat_children = yhat.chn
                    yhat_edge_probs = yhat(x)[x_i,:]
                    for i,yhat_child in enumerate(yhat_children):
                        p_yhat_child = yhat_edge_probs[i]*yhat_prob
                        Q.push(p_yhat_child,yhat_child)
            
            pred_list.append(y_best)
            EU_list.append(EU_best.item())
        return pred_list, EU_list
            
    """
    RBOP
    note that x is a batch of instances (pytorch currently only supports batch-wise prediction)
    """
    def predict_rbop(self, x, loss, params, epsilon):
        pred_list = []
        EU_list = []
        for x_i in range(x.shape[0]):            
            Q = PriorityList()
            Q.push(1.,self.root)
            K = PriorityList()
            
            yhat_eps = self.root.y
            EU_eps = self.EU(self.root.y,1.,loss,params)
            
            while not Q.is_empty():
                yhat_prob, yhat = Q.pop()
                yhat_prob = 1.-yhat_prob
                if len(yhat.chn) == 0:
                    K.remove_all()
                    break
                yhat_children = yhat.chn
                yhat_edge_probs = yhat(x)[x_i,:]
                inserted = []
                for i,yhat_child in enumerate(yhat_children):
                    p_yhat_child = yhat_edge_probs[i]*yhat_prob
                    if p_yhat_child >= epsilon:
                        Q.push(p_yhat_child,yhat_child)
                        inserted.append(True)
                        eu_yhat_child = self.EU(yhat_child.y,p_yhat_child,loss,params)
                        if eu_yhat_child > EU_eps:
                            yhat_eps = yhat_child.y
                            EU_eps = eu_yhat_child
                    else:
                        inserted.append(False)
                if True not in inserted:
                    K.push(yhat_prob,yhat)
                    
            while not K.is_empty():
                yhat_prob_p, yhat = K.pop()
                yhat_prob_p = 1.-yhat_prob_p
                while(len(yhat.y) > 1):
                    opt_p = 0
                    yhat_edge_probs = yhat(x)[x_i,:]
                    for i,c in enumerate(yhat.chn):
                        c_p = yhat_edge_probs[i]*yhat_prob_p
                        if c_p > opt_p:
                            opt_p = c_p
                            yhat = c
                    EU_yhat = self.EU(yhat.y,opt_p,loss,params)
                    if EU_yhat > EU_eps:
                        yhat_eps = yhat.y
                        EU_eps = EU_yhat
                    yhat_prob_p = opt_p
            pred_list.append(yhat_eps)
            EU_list.append(EU_eps.item())
        return pred_list, EU_list
    
"""
Hierarchical softmax neural network for image data
"""
class HMCC(nn.Module):
    def __init__(self,struct,gpu,patience,vgg,ds,trdl=None,vldl=None):
        super(HMCC, self).__init__()
        # register struct, dataloader object, etc.
        self.struct = deepcopy(struct)
        self.trdl = trdl
        self.vldl = vldl
        if gpu:
            self.dtype = torch.cuda.FloatTensor
        else:
            self.dtype = torch.FloatTensor
        self.gpu = gpu
        self.patience = patience
        self.vgg = vgg
        self.dataset = ds

        # feature extraction part
        if vgg:
            self.features = nn.Sequential(
                *list(models.vgg16_bn(pretrained=True).features.children())
            )
            self.ft_size = 25088
            #self.ft_size = 128
        elif "PROTEIN" in ds:
            self.embedding = nn.Conv1d(26,8,1)
            self.ft_scale_cnn = nn.ModuleList([])
            for scale in [3,5,7,9,11]:
                self.ft_scale_cnn.append(nn.Sequential(
                    nn.Conv1d(8,16,scale,padding=(scale-1)//2),
                    nn.Conv1d(16,32,scale,padding=(scale-1)//2),
                    nn.BatchNorm1d(32),
                    nn.ReLU(),
                    nn.MaxPool1d(2,3),
                    nn.Conv1d(32,64,scale,padding=(scale-1)//2),
                    nn.Conv1d(64,128,scale,padding=(scale-1)//2),
                    nn.BatchNorm1d(128),
                    nn.ReLU(),
                    nn.MaxPool1d(2,3),
                    nn.Conv1d(128,256,scale,padding=(scale-1)//2),
                    nn.BatchNorm1d(256),
                    nn.ReLU(),
                    nn.MaxPool1d(2),
                ))
            if gpu:
                self.embedding.cuda()
                self.ft_scale_cnn.cuda()
            out_embedding = self.embedding(torch.randn(32,26,4911).type(self.dtype))
            out_scale_cnn = []
            for cls in self.ft_scale_cnn:
                print(cls(out_embedding).shape)
                out_scale_cnn.append(cls(out_embedding))
            out_scale_cnn = torch.cat(out_scale_cnn,1)
            out_scale_cnn = out_scale_cnn.transpose(1,2)
            self.ft_size_scale_cnn = out_scale_cnn.shape
            print("SCALE CNN SHAPE = {0}".format(self.ft_size_scale_cnn))
            self.rnn = nn.LSTM(self.ft_size_scale_cnn[2], 128, 1, batch_first=True)
            if gpu:
                self.rnn.cuda()
            out_rnn = self.rnn(out_scale_cnn)[0]
            out_rnn = out_rnn.transpose(1,2)
            self.ft_size_rnn = out_rnn.shape
            print("SCALE RNN SHAPE = {0}".format(self.ft_size_rnn))
            self.ft_size = self.ft_size_rnn[1]*self.ft_size_rnn[2]
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

        # hierarchical tree classifier part
        self.hnet = HNet(self.ft_size, deepcopy(struct), dtype=self.dtype)

    def forward(self, x, l=None, train=True):
        if "PROTEIN" in self.dataset:
            x = self.embedding(x)
            x_scale_cnn = []
            for cls in self.ft_scale_cnn:
                x_scale_cnn.append(cls(x))
            x_scale_cnn = torch.cat(x_scale_cnn,1)
            x_scale_cnn = x_scale_cnn.transpose(1,2)
            x_rnn = self.rnn(x_scale_cnn)[0]
            x_rnn = x_rnn.transpose(1,2)
            if not x_rnn.is_contiguous():
                x_rnn = x_rnn.contiguous()
            x = x_rnn.view(-1,self.ft_size)
            x = self.hnet(x, l=l, train=train)
            return x
        else:
            x = self.features(x)
            x = x.view(-1, self.ft_size)
            x = self.hnet(x, l=l, train=train)
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

        if not ft:
            if "PROTEIN" in self.dataset:
                optimizer = optim.Adam(self.parameters(), lr=lr)
            else:
                optimizer = optim.Adam(self.hnet.parameters(), lr=lr)
        else:
            optimizer = optim.Adam(self.parameters(), lr=lr)

        # intialize variables for patience mechanics
        min_val_loss = sys.maxsize
        patience_cnt = 0
        for epoch in range(ne):  # loop over the dataset multiple times
            train_running_loss = 0.0
            train_time = 0.0
            val_running_loss = 0.0
            val_acc = 0.0

            # loop over training data
            counter_train = 0
            for i, data in enumerate(self.trdl, 0):
                inputs, labels = data
                inputs, labels = Variable(inputs.type(self.dtype)), Variable(labels.type(self.dtype), requires_grad=False)
                
                # obtain "plain" targets
                targets = labels.cpu().data.numpy()
                targs_to_class = np.argmax(targets,axis=1)
                targs_to_list = [c+1 for c in targs_to_class]
                
                # set model to training mode
                self.train()

                # zero the parameter gradients
                optimizer.zero_grad()
                
                # obtain loss, and do backprop
                start_time = time.time()
                losses = self(inputs,targs_to_list)
                loss = torch.mean(torch.stack(losses))
                loss.backward()
                optimizer.step()
                train_time += time.time()-start_time

                # print statistics
                train_running_loss += loss.item()
                counter_train += 1

            # loop over validation data
            counter_val = 0
            for i, data in enumerate(self.vldl, 0):

                # set model to evaluation mode and
                self.eval()

                inputs, labels = data
                inputs, labels = Variable(inputs.type(self.dtype)), Variable(labels.type(self.dtype))
                
                # obtain "plain" targets
                targets = labels.cpu().data.numpy()
                targs_to_class = np.argmax(targets,axis=1)
                targs_to_list = [c+1 for c in targs_to_class]
                
                # calculate losses and accuracy
                losses = self(inputs,targs_to_list)
                loss = torch.mean(torch.stack(losses))
                val_running_loss += loss.item()
                val_acc += calculate_accuracy(self(inputs,train=False), labels)

                counter_val += 1
            print("EPOCH {0}: lossTr={1:.3f}    timeTr={2:4f}    lossVal={3:.3f}    accVal={4:.3f}".format(epoch + 1, train_running_loss/counter_train, train_time/counter_train, val_running_loss/counter_val, val_acc/counter_val))
            self.save_state("inter_models/{0}/HIERARCHICAL_MCC_{1}_{2}".format(self.dataset,lr,epoch+1))

            # check if early stopping applies
            valLoss = round(val_running_loss/counter_val,4)
            if valLoss < min_val_loss:
                min_val_loss = valLoss
                patience_cnt = 0
            else:
                patience_cnt += 1
                if patience_cnt == self.patience:
                    print("[info] early stopping applied after exceeding patience counter of {0}".format(self.patience))
                    break
                    
        print("done!")
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

