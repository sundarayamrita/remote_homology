from operator import pos
import torch
from torch import nn
#from torchvision import transforms
from torch.utils.data import TensorDataset, DataLoader, random_split
#from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np
import pandas as pd
import argparse
from pathlib import Path
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import MinMaxScaler
from keras.utils import to_categorical
from Bio import SeqIO
from Bio.SeqUtils.ProtParam import ProteinAnalysis
import csv
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()

        # Convolution 1
        self.cnn1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=2)
        self.relu1 = nn.ReLU()

        # Max pool 1
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)

        # Convolution 2
        self.cnn2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=2)
        self.relu2 = nn.ReLU()
        # Max pool 2
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)

        # Fully connected 1 (readout)
        self.fc1 = nn.Linear(32 * 232 * 2, 2)
        # self.fc1 = nn.Linear(32 * 232, 1)

    def forward(self, x):
        # Convolution 1
        out = self.cnn1(x)
        out = self.relu1(out)

        # Max pool 1
        out = self.maxpool1(out)

        # Convolution 2 
        out = self.cnn2(out)
        out = self.relu2(out)

        # Max pool 2 
        out = self.maxpool2(out)

        # Resize
        # Original size: (100, 32, 7, 7)
        # out.size(0): 100
        # New out size: (100, 32*7*7)
        out = out.view(out.size(0), -1)

        # Linear function (readout)
        out = self.fc1(out)

        return out

parser = argparse.ArgumentParser()
parser.add_argument("-data_dir", help = "dataset folder", type = str)
args = parser.parse_args()
if not args.data_dir:
    print("Dataset directory does not exist")
    exit()
else:
    dataset = Path(args.data_dir)

#pos_x = np.load('pos-train.c.1.1.npy')
#neg_x = np.load('neg-train.c.1.1.npy')
for file in sorted(dataset.glob('*.txt')):
    if 'pos-train.c.1.1' in file.name:
        pos_x = []
        with open(file, 'r') as f:
            reader = csv.reader(f,delimiter='\n')
            for row in reader:
                if(">seq" not in row):
                    pos_x.append(row)
    if 'neg-train.c.1.1' in file.name:
        neg_x = []
        with open(file, 'r') as f:
            reader = csv.reader(f,delimiter='\n')
            for row in reader:
                if(">seq" not in row):
                    neg_x.append(row)
    if 'pos-test.c.1.1' in file.name:
        pos_test_x = []
        with open(file, 'r') as f:
            reader = csv.reader(f,delimiter='\n')
            for row in reader:
                if(">seq" not in row):
                    pos_test_x.append(row)
    if 'neg-test.c.1.1' in file.name:
        neg_test_x = []
        with open(file, 'r') as f:
            reader = csv.reader(f,delimiter='\n')
            for row in reader:
                if(">seq" not in row):
                    neg_test_x.append(row)


pos_x_one= []
coding_record = []
protein_names = []
coding = ['A','R','N','D','C','Q','E','G','H', 'I','L','K','M','F','P','S','T','W','Y','V']
aaVal = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                           1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]

for record in pos_x:
            sequence = record[0]
            for aa in sequence:
                aa = aa.upper()
                if aa not in coding:
                    print("goes here")
                    continue
                coding.append(coding.index(aa))
            coding_record.append(coding)


print("Y is ",coding_record)