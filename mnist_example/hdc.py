import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.datasets import MNIST

# Note: this example requires the torchmetrics library: https://torchmetrics.readthedocs.io
import torchmetrics
from tqdm import tqdm

import torchhd
from torchhd.models import Centroid
from torchhd import embeddings

import math

import numpy as np
import random
import sys
import os

import pathlib
path = str(pathlib.Path(__file__).parent.resolve())

np.set_printoptions(threshold=sys.maxsize)
torch.set_printoptions(threshold=sys.maxsize)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using {} device".format(device))

DIMENSIONS = 1000
IMG_SIZE = 28
NUM_LEVELS = 256
c = int(math.floor(DIMENSIONS/NUM_LEVELS))
r = int (DIMENSIONS%NUM_LEVELS)
print_flag = 0 
countTimer = 100
BATCH_SIZE = 1  # for GPUs with enough memory we can process multiple images at ones
a = []
b = []
transform = torchvision.transforms.PILToTensor()

train_ds = MNIST(path+"/data", train=True, transform=transform, download=True)   #, transform=transform
train_ld = torch.utils.data.DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

test_ds = MNIST(path+"/data", train=False, transform=transform, download=True) #, transform=transform
test_ld = torch.utils.data.DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)


class LFSR:
    def __init__(self, init, XORs):
        self.register = init
        self.XORs = XORs

    def shift(self):
        feedback = self.register[len(self.register)-1]
        for XOR in self.XORs:
            self.register[XOR-1] ^= feedback
        self.register = [feedback]+ self.register[0:len(self.register)-1]
        return self.register

    def generate_sequence(self, length):
        sequence = []
        init = self.register
        for i in range(length):
            reg = self.shift()
            bipolarReg = [-1 if x==0 else x for x in reg]
            if (init == reg ):
                #print(i+1)
                sequence.append(bipolarReg[::-1].copy())
                return sequence
            sequence.append(bipolarReg[::-1].copy())
        return sequence

class Encoder(nn.Module):
    def __init__(self, out_features, size, levels):
        super(Encoder, self).__init__()
        self.flatten = torch.nn.Flatten()
        self.position = embeddings.Random(size * size, out_features)
        self.value = embeddings.Level(levels, out_features, high=256)
        #### my levels
        levels = []
        for number in range(NUM_LEVELS):
            #print(number, end = ", ")
            my_list = []
            if number == 0:
                my_list = [-1]*(DIMENSIONS)
            elif number == NUM_LEVELS-1:
                my_list = [1]*(DIMENSIONS)
            else:
                #my_list = [-1]*(DIMENSIONS-(number*c)) + [1]*(number*c)
                mm = [-1]*(NUM_LEVELS-(number)) + [1]*(number)
                for i in range(c):
                    my_list = mm + my_list
                if  r!= 0:
                    my_list = mm[-r::] + my_list
            #st (my_list)
            #my_list = my_list[::-1]
            # if number == 230:
            #     print("\n")
            #     st (my_list)
            
            levels.append(my_list)
        #print("len (mm[r::]) \n",len(mm[-r::]), "len (mm[::r])",len (mm[::r]))
        #print(r)
        #print(c)
        arr = np.array(levels)
        #print("ID_LEVELS")
        #print(arr)
        #np.save("./mem/mnist_levels.npy", arr)
        tArr = torch.nn.Parameter(torchhd.MAPTensor(torch.from_numpy(arr).float()))
        self.value.weight = tArr.float()

        self.init_num = random.randint(1, 2**out_features)
        #init_num = 4005700534675144948017520162097884868995542063410121040524186571954440390130004122707793153004062360041636382462054526905859450700091502234942087738732084346636139513840569559345488256204180986892293042541798471825399139750542421374166982260754066786698152028283234760575400288349132987005389102090211
        XORs_num = random.randint(2**(out_features-1), 2**out_features)
        #XORs_num = 8304480432096973745855277495005820904147225062766752395232516640190584805313156405292672642268352808496341593728537662541323806162655398579460968638548439233518083677818779822153866372461399193020495636799816940385452684276171359521470779611083416341728702669855951160018696384825076986035152844977163
        init = [eval(i) for i in [*bin(self.init_num)[2:]]]
        init.extend([0] * (out_features - len(init)))
        self.XORs = [i for i, x in enumerate(reversed([*bin(XORs_num)[2:]])) if x == '1']
        lfsr = LFSR(init, self.XORs)
        sequence_length = size * size
        self.generated_sequence = lfsr.generate_sequence(sequence_length)
        arr = np.array(self.generated_sequence)
        #print(arr)
        #np.save("./mem/mnist_bv.npy", arr)
        tArr = torch.nn.Parameter(torchhd.MAPTensor(torch.from_numpy(arr).float()))
        self.position.weight = tArr.float()
    
        #genConfig (XORs, init_num, DIMENSIONS)
        #write_memory(self.XORs, self.init_num, self.generated_sequence, NUM_LEVELS, DIMENSIONS)

    def forward(self, x):
        x = self.flatten(x)
        sample_hv = torchhd.bind(self.position.weight, self.value(x))
        sample_hv = torchhd.multiset(sample_hv)
        positive = torch.tensor(1.0, dtype=sample_hv.dtype, device=sample_hv.device)
        negative = torch.tensor(-1.0, dtype=sample_hv.dtype, device=sample_hv.device)
        return torch.where(sample_hv > 0, positive, negative)

encode = Encoder(DIMENSIONS, IMG_SIZE, NUM_LEVELS)
encode = encode.to(device)

num_classes = len(train_ds.classes)
model = Centroid(DIMENSIONS, num_classes)
model = model.to(device)

def train():
    with torch.no_grad():
        for samples, labels in tqdm(train_ld, desc="Training"):
            samples = samples.to(device)
            labels = labels.to(device)

            samples_hv = encode(samples)
            model.add(samples_hv, labels)

def test():
    accuracy = torchmetrics.Accuracy("multiclass", num_classes=num_classes)
    model.normalize(quantize=True)
    with torch.no_grad():
        print_flag = 1
        count = 0
        for samples, labels in tqdm(test_ld, desc="Testing"):
            count = count + 1
            if count %850 == 0:
                print_flag = 1
            else:
                print_flag = 0
            samples = samples.to(device)
            samples_hv = encode(samples)
            outputs = model(samples_hv, dot=True)
            #if print_flag == 1:
            accuracy.update(outputs.cpu(), labels)

    print(f"Testing accuracy of {(accuracy.compute().item() * 100):.3f}%")

    torch.save(model.weight,                path+"/model/chvs.pt")
    torch.save(encode.init_num,             path+"/model/init_num.pt")
    torch.save(encode.XORs,                 path+"/model/xors.pt")
    torch.save(encode.generated_sequence,   path+"/model/sequence.pt")

#ls = class_sparsity (model.weight)
