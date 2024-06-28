import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.datasets import MNIST

# Note: this example requires the torchmetrics library: https://torchmetrics.readthedocs.io
import torchmetrics
from tqdm import tqdm

import torchhd
from torchhd import Centroid
from torchhd import embeddings

from torchvision import datasets
from torchvision.transforms import transforms,ToTensor
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import confusion_matrix

import numpy as np
import random
import sys
import os
torch.set_printoptions(threshold=sys.maxsize)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using {} device".format(device))

BATCH_SIZE = 1

transform = torchvision.transforms.ToTensor()

train_ds = MNIST("../data", train=True, transform=transform, download=True)
train_ld = torch.utils.data.DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
test_ds = MNIST("../data", train=False, transform=transform)
test_ld = torch.utils.data.DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

DIMENSIONS = [1000, 10000]
IMG_SIZE = 28
CHANNELS = 1
num_classes = 10
NUM_LEVELS = 10000
#NUM_STANDARD: percentage of standard training samples, rest are used for adaptive
highest = 0
quantize = [True]


###### =====================================RANDOM PRO=============================================

class Encoder_rand(nn.Module):
    def __init__(self, out_features, size, levels):
        super(Encoder, self).__init__()
        self.flatten = torch.nn.Flatten()
        self.project = embeddings.Sinusoid(size * size, out_features)

    def forward(self, x):
        x = self.flatten(x)
        sample_hv = self.project(x)
        sample_hv = torchhd.multiset(sample_hv)
        return torchhd.hard_quantize(sample_hv)

###### =====================================BASE LEVEL=============================================

class Encoder_base(nn.Module):
    def __init__(self, out_features, size, levels):
        super(Encoder, self).__init__()
        self.flatten = torch.nn.Flatten()
        self.position = embeddings.Random(size * size, out_features)
        self.value = embeddings.Level(levels, out_features)

    def forward(self, x):
        x = self.flatten(x)
        sample_hv = torchhd.bind(self.position.weight, self.value(x))
        sample_hv = torchhd.multiset(sample_hv)
        return torchhd.hard_quantize(sample_hv)




encoders = Encoder_rand,  Encoder_base
for Encoder in encoders:
    for d in DIMENSIONS:
        for q in quantize:
            highest = 0
            i = 0
            encode = Encoder(d, IMG_SIZE, NUM_LEVELS)
            print(d," --- " ,encode)
            encode = encode.to(device)


            num_classes = len(train_ds.classes)
            model = Centroid(d, num_classes)
            model = model.to(device)



            ###### =====================================LOAD the model=============================================
            classes = []
            modelload = "MNISTmodels/MNIST"+"_"+ str(Encoder)[17:-2]+"_quantize_"+str(q)+"_"+str(d)
            modelload1 = "./"+modelload+".pt"
            modelood = Centroid(d, num_classes)
            weights = torch.load( modelload1 , map_location=torch.device('cpu'))
            #print(weights[0])
            modelood.weight = weights
            #modelood.normalize(quantize=q)
            #print ("modelood.weight after normalize  ", modelood.weight[4])
            #classes.append(modelood.weight.cpu().detach().numpy())
            #print ("classes.  ", classes[0], len(classes) , len(classes[0]))
            k = modelood.weight
            #print ("k.  ", k[0], len(k) , len(k[0]))
            ls = []
            counter = 0
            for j in range(len(k[0])):
                m = k[0][j]
                f = 0
                for i in range(len(k)):
                    #print(k[i][j])
                    if m != k[i][j]:
                        f = 1
                if f == 1:
                    ls.append(j)
            print(len(ls))#, ls)
            os.system('touch connector'+str(d)+"TO"+str(len(ls))+'.vhd')
            f = open('connector'+str(d)+"TO"+str(len(ls))+'.vhd', "w")
            f.write("LIBRARY IEEE; \nUSE IEEE.STD_LOGIC_1164.ALL; \nUSE IEEE.NUMERIC_STD.ALL; \n  \nENTITY connector IS \n\tPORT ( \n\t\tinput         : IN  STD_LOGIC_VECTOR ("+ str(d - 1)+" DOWNTO 0); \n\t\tpruneoutput        : OUT  STD_LOGIC_VECTOR ("+ str(len(ls)- 1)+" DOWNTO 0)      \n\t);\nEND ENTITY connector"+str(d)+"TO"+str(len(ls))+";\n\nARCHITECTURE behavioral OF connector"+str(d)+"TO"+str(len(ls))+"  IS\nBEGIN\n")
            for i in range(d):
                if i in ls:
                    f.write("pruneoutput("+str(counter)+") <= input("+str(i)+");\n")
                    counter = counter + 1
            f.write('\nEND ARCHITECTURE behavioral;')
            f.close()
