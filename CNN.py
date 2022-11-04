import argparse
import torch
import torch.nn as nn
from torch.utils.data import random_split, DataLoader
import torch.nn.functional as F
from torchvision.datasets import CIFAR10
from torchvision import transforms
import pytorch_lightning as pl
from torchmetrics import Accuracy
import os

class CIFAR10DataModule(pl.LightningDataModule):
    def __init__(self,mb=4,pin_memory=False,num_workers=0,val_n=5000):
        super().__init__()
        self.mb = mb
        self.pin_memory = pin_memory
        self.num_workers = num_workers
        self.val_n = val_n 
        self.transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    def prepare_data(self):
        # download data if not downloaded
        CIFAR10(root='./data', train=True, download=True)
    def setup(self,stage=None):
        # make splits, create Dataset objects
        if stage == 'fit' or stage is None:
            # Load train and val
            trainvalset = CIFAR10(root='./data',train=True,
                    transform=self.transform)
            self.trainset,self.valset = random_split(trainvalset,
                    [len(trainvalset)-self.val_n,self.val_n])
        if stage == 'test' or stage is None:
            # Load test set
            self.testset = CIFAR10(root='./data',train=False,
                    transform=self.transform)
    def train_dataloader(self):
        return DataLoader(self.trainset, shuffle=True, batch_size=self.mb, 
                pin_memory=self.pin_memory, num_workers=self.num_workers)
    def val_dataloader(self):
        return DataLoader(self.valset, batch_size=self.mb, 
                pin_memory=self.pin_memory, num_workers=self.num_workers)
    def test_dataloader(self):
        return DataLoader(self.testset, batch_size=self.mb, 
                pin_memory=self.pin_memory, num_workers=self.num_workers)

class CNN(pl.LightningModule):
    def __init__(self,lr=0.1, opt='sgd'):
        super().__init__()
        self.accuracy = Accuracy()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv1 = nn.Conv2d(3, 64, 3, 1, 1)
        self.conv2 = nn.Conv2d(64, 64, 3, 1, 1)

        self.conv3 = nn.Conv2d(64, 128, 3, 1, 1)
        self.conv4 = nn.Conv2d(128, 128, 3, 1, 1)

        self.conv5 = nn.Conv2d(128, 256, 3, 1, 1)
        self.conv6 = nn.Conv2d(256, 256, 3, 1, 1)

        self.conv7 = nn.Conv2d(256, 512, 3, 1, 1)
        self.conv8 = nn.Conv2d(512, 512, 3, 1, 1)

        self.dropout = nn.Dropout2d(0.5)

        self.fc1 = nn.Linear(512 * 4 * 4, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, 10)
        self.lr = lr
        self.opt = opt
    def forward(self, x):
        sizex,sizey,sizez,_ = x.size()
        x = x.view(sizex, sizey, sizez, -1)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.pool(x)
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = self.pool(x)
        x = F.relu(self.conv7(x))
        x = F.relu(self.conv8(x))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x
    def eval_batch(self, batch, batch_idx):
        # Make predictions
        x, y = batch
        y_pred = self(x)
        # Evaluate predictions
        loss = F.cross_entropy(y_pred, y)
        acc = self.accuracy(y_pred, y)
        return loss, acc
    def training_step(self, batch, batch_idx):
        loss,acc = self.eval_batch(batch,batch_idx)
        x,y = batch
        y_pred = self(x)
        self.log('train_loss', loss)
        self.log('train_acc', acc)
        return loss
    def validation_step(self, batch, batch_idx):
        loss,acc = self.eval_batch(batch,batch_idx)
        self.log('val_loss', loss)
        self.log('val_acc', acc, prog_bar=True)
    def test_step(self, batch, batch_idx):
        loss,acc = self.eval_batch(batch,batch_idx)
        self.log('test_loss', loss)
        self.log('test_acc', acc)
    def configure_optimizers(self):
        if   self.opt == 'adadelta':
            return torch.optim.Adadelta(self.parameters(), lr=self.lr)
        elif self.opt == 'adagrad':
            return torch.optim.Adagrad(self.parameters(), lr=self.lr)
        elif self.opt == 'adam':
            return torch.optim.Adam(self.parameters(), lr=self.lr)
        elif self.opt == 'rmsprop':
            return torch.optim.RMSprop(self.parameters(), lr=self.lr)
        elif self.opt == 'sgd':
            return torch.optim.SGD(self.parameters(), lr=self.lr)

def parse_all_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("-opt",type=str,\
            help='The optimizer: "adadelta", "adagrad", "adam", "rmsprop", "sgd" (string) [default: "sgd"]',\
            default="sgd")
    parser.add_argument("-lr", type=float,\
            help='The learning rate (float) [default: 0.1]',\
            default=0.1)
    parser.add_argument("-mb", type=int,\
            help='The minibatch size (int) [default: 4]',\
            default=4)
    parser.add_argument("-epochs",type=int,\
            help="The number of training epochs (int) [default: 100]",\
            default=100)

    return parser.parse_args()

def main():
    args = parse_all_args()
    data  = CIFAR10DataModule(mb=args.mb, num_workers=2)
    model = CNN(lr=args.lr, opt=args.opt)
    trainer = pl.Trainer(max_epochs=args.epochs, gpus=1)
    trainer.fit(model,data)
    # trainer.test(model,datamodule=data)

if __name__ == '__main__':
    main()