import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import argparse
import sys
import numpy as np
import re

# takes comma seperated str of nunits x mlayers (e.g. 128x2,16x1,128x2)
# returns list of dimensions to construct layers with
def parse_nn_size(dim_str: str) -> list:
    splits = dim_str.split(',')
    matches = [re.search(r'(\d+)x(\d+)', dim) for dim in splits]
    
    dims = []
    for match in matches:
        dims += [int(match[1]) for i in range(int(match[2]))]
    return dims

# takes in list of layer dimensions
# returns list of linear layers with corresponding dimensions
def create_layers(dims: list) -> list:
    # receive list of layer dimensions
    # create a layer for each list item and append to list
    layers = [nn.Linear(dims[i], dims[i+1]) for i in range(len(dims[:-1]))]
    # layers.append(nn.Linear(dims[-1], C)) 
    return layers

class MNISTDataset(Dataset):
    def __init__(self, x_file, y_file):
        self.inputs = np.load(x_file).astype(np.float32)
        self.targets = np.load(y_file).astype(np.float32)
        return
    
    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index) -> tuple:
        if torch.is_tensor(index):
            index = index.tolist()
        return self.inputs[index, :], self.targets[index]
    
    def D(self):
        _, in_features = self.inputs.shape
        return in_features

class DeepNeuralNet(nn.Module):
    def __init__(self, dims: str, C: int, in_features: int, f1: str):
        # init super class
        super(DeepNeuralNet, self).__init__()
        
        # create linear layers
        dim_list = [in_features] + parse_nn_size(dims) + [C]
        layers_list = create_layers(dim_list)
        self.layers = nn.ModuleList(layers_list)

        # set activation function
        if   f1 == 'relu':
            self.activation = nn.ReLU()
        elif f1 == 'tanh':
            self.activation = nn.Tanh()
        elif f1 == 'sigmoid':
            self.activation = nn.Sigmoid()
        
        # print model parameters
        for name, param in self.named_parameters():
            print(name,param.data.shape)
    
    def forward(self, x):
        # perform a forward pass through each layer linearly
        for l in self.layers[:-1]:
            x = self.activation(l(x))
        # skip using an activation function on the last layer
        x = self.layers[-1](x)
        return x
    
    def get_weights(self):
        return [l.weights for l in self.layers]

def parse_all_args():
    parser = argparse.ArgumentParser()

    # positional args
    parser.add_argument("C", type=int, help="The number of classes (int)")
    parser.add_argument("train_x",help="The training set input data (npz)")
    parser.add_argument("train_y",help="The training set target data (npz)")
    parser.add_argument("dev_x",help="The development set input data (npz)")
    parser.add_argument("dev_y",help="The development set target data (npz)")

    # optional args
    parser.add_argument("-f1",type=str,\
            help='The hidden activation function: "relu" or "tanh" or "sigmoid" (string) [default: "relu"]',\
            default="relu")
    parser.add_argument("-opt",type=str,\
            help='The optimizer: "adadelta", "adagrad", "adam", "rmsprop", "sgd" (string) [default: "adam"]',\
            default="adam")
    parser.add_argument("-L", type=str,\
            help='A comma delimited list of nunits by nlayers specifiers (see assignment pdf) (string) [default: "32x1"]',\
            default="32x1")
    parser.add_argument("-lr", type=float,\
            help='The learning rate (float) [default: 0.1]',\
            default=0.1)
    parser.add_argument("-mb", type=int,\
            help='The minibatch size (int) [default: 32]',\
            default=32)
    parser.add_argument("-report_freq", type=int,\
            help='Dev performance is reported every report_freq updates (int) [default: 128]',\
            default=128)
    parser.add_argument("-epochs",type=int,\
            help="The number of training epochs (int) [default: 100]",\
            default=100)

    return parser.parse_args()

def train(model, train_loader, dev_loader, args):

    criterion = nn.CrossEntropyLoss()

    # select an optimizer from the commandline argument
    if   args.opt == 'adadelta':
        optimizer = torch.optim.Adadelta(model.parameters(), lr=args.lr)
    elif args.opt == 'adagrad':
        optimizer = torch.optim.Adagrad(model.parameters(), lr=args.lr)
    elif args.opt == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    elif args.opt == 'rmsprop':
        optimizer = torch.optim.RMSprop(model.parameters(), lr=args.lr)
    elif args.opt == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
        
    val_counter = args.report_freq

    for epoch in range(args.epochs):        
        for update, (mb_x,mb_y) in enumerate(train_loader):           
        
            y_pred = model(mb_x)
            
            loss = criterion(y_pred, mb_y.long())
            
            # take an optimizer step on each minibatch
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # eval model with devset every report_freq minibatch steps
            val_counter -= 1
            if val_counter <= 0:
                val_counter = args.report_freq
                
                true_pos = 0
                total = 0
                for update, (d_mb_x,d_mb_y) in enumerate(dev_loader):
                    dev_y_pred = model(d_mb_x)
                    _, dev_predicted = torch.max(dev_y_pred, 1)
                    true_pos += torch.sum(dev_predicted == d_mb_y)
                    total += len(d_mb_y)
                dev_acc = true_pos / total
                print("dev acc = %.3f" % (dev_acc*100))

def main(argv):
    args = parse_all_args()

    # Create datasets for train/dev
    trainDataset = MNISTDataset(args.train_x, args.train_y)
    devDataset   = MNISTDataset(args.dev_x, args.dev_y)

    # Create dataloaders for train/dev
    trainDataLoader = DataLoader(trainDataset,\
        shuffle=True, drop_last=False, batch_size=args.mb)
    devDataLoader   = DataLoader(devDataset,\
        shuffle=False, drop_last=False, batch_size=args.mb)

    # create model
    in_features = trainDataset.D()
    model = DeepNeuralNet(args.L, args.C, in_features, args.f1)
    
    # Train model on train/dev sets
    train(model, trainDataLoader, devDataLoader, args)

if __name__ == "__main__":
    main(sys.argv)
