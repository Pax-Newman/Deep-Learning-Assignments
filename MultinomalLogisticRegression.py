import torch
import argparse
import sys
import numpy as np

class MultinomialLogReg(torch.nn.Module):
    def __init__(self, C, in_features):
        super(MultinomialLogReg, self).__init__()
        self.weights = torch.nn.Parameter(torch.zeros((C, in_features)))
        self.biases = torch.nn.Parameter(torch.zeros(C))
    
    def forward(self, x):
        y_pred = (x @ self.weights.T) + self.biases
        return y_pred
    
    def get_weights(self):
        return self.weights

def parse_all_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("C", type=int, help="The number of classes for classification")
    
    parser.add_argument("train_x",help="The training set input data (npz)")
    parser.add_argument("train_y",help="The training set target data (npz)")
    parser.add_argument("dev_x",help="The development set input data (npz)")
    parser.add_argument("dev_y",help="The development set target data (npz)")

    parser.add_argument("-lr",type=float,\
            help="The learning rate (float) [default: 0.1]",default=0.1)
    parser.add_argument("-epochs",type=int,\
            help="The number of training epochs (int) [default: 100]",\
            default=100)
    parser.add_argument("-lambda", type=float, help="The scaling coefficient (float) [default: 0.0]", default=0.0)

    return parser.parse_args()

def train(model,train_x,train_y,dev_x,dev_y,args):
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    for epoch in range(args.epochs):
        y_pred = model(train_x)
        norm = model.weights.pow(2).sum()
        loss = criterion(y_pred, train_y.long()) + getattr(args, 'lambda') * norm
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        _, train_predicted = torch.max(y_pred.data, 1)
        train_acc = torch.sum(train_predicted == train_y) / len(train_y)
        
        dev_y_pred = model(dev_x)
        _, dev_predicted = torch.max(dev_y_pred, 1)
        dev_acc = torch.sum(dev_predicted == dev_y) / len(dev_y)
        
        print("train acc = %.3f, dev acc = %.3f" % (train_acc*100,dev_acc*100))

def main(argv):
    args = parse_all_args()

    train_x = torch.from_numpy(np.load(args.train_x).astype(np.float32))
    train_y = torch.from_numpy(np.load(args.train_y).astype(np.float32))
    dev_x   = torch.from_numpy(np.load(args.dev_x).astype(np.float32))
    dev_y   = torch.from_numpy(np.load(args.dev_y).astype(np.float32))

    _, in_features = train_x.size()
    model = MultinomialLogReg(args.C, in_features)
    
    train(model,train_x,train_y,dev_x,dev_y,args)

if __name__ == "__main__":
    main(sys.argv)
