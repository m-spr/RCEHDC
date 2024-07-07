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

import sympy as sp
from sympy import Eq, Symbol, solve, N
import math

import numpy as np
import tensorflow as tf
import random
import sys
np.set_printoptions(threshold=sys.maxsize)
torch.set_printoptions(threshold=sys.maxsize)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using {} device".format(device))

DIMENSIONS = 1000
IMG_SIZE = 28
NUM_LEVELS = 1000
c = 1
print_flag = 0 
countTimer = 100
BATCH_SIZE = 1  # for GPUs with enough memory we can process multiple images at ones
a = []
b = []
transform = torchvision.transforms.ToTensor()

train_ds = MNIST("../data", train=True, transform=transform, download=True)
train_ld = torch.utils.data.DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

test_ds = MNIST("../data", train=False, transform=transform, download=True)
test_ld = torch.utils.data.DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

def memMinimizer (d, f):
    n = 2
    k = []
    zeros = []
    memSize = []
    popCountSize = []
    memOverhead = []
    while 2**n < f:
        k.append(math.ceil(d/(2**n)))
        zeros.append((k[-1]*(2**n))-d)
        memSize.append(n)
        popCountSize.append((k[-1]*n))
        memOverhead.append(zeros[-1]+popCountSize[-1])
        n = n + 1
    return memSize[memOverhead.index(min(memOverhead))], k[memOverhead.index(min(memOverhead))]

def genConfig (XORs, init_num, DIMENSIONS):
    strXors = ""
    strinit = ""
    #print (init)
    for i in range(DIMENSIONS):
        if i in XORs :
              strXors = strXors + '1'
        else:
              strXors = strXors + '0'
    init = [eval(i) for i in [*bin(init_num)[2:]]]
    for i in (init):
        strinit = strinit + str(i)
    for i in range (DIMENSIONS - len(init)):
        strinit =  strinit +'0'
    with open('Xili_HDCMem/lfsrConfig.txt', 'w') as output:
        output.write("XORs\n")
        output.write(str(strXors[::-1]))
        output.write("\n")
        output.write("init\n")
        output.write(strinit[::-1])
#def checkOrtogonal (x):
#    for i in x:
#        for j in x:
#            if i
def config (DIMENSIONS, featureSize, NUM_LEVELS, classes):
    pixbit = math.ceil(math.log2(NUM_LEVELS))
    d = DIMENSIONS
    lgf  = math.ceil(math.log2(featureSize))
    c   = classes
    f = featureSize
    n , adI = memMinimizer (DIMENSIONS, featureSize)
    adz = 2**(math.ceil(math.log2(adI)))-adI
    zComp =  2**(math.ceil(math.log2(classes)))-classes
    lgCn =  math.ceil(math.log2(classes))
    logn = math.ceil(math.log2(adI))
    x = math.ceil(DIMENSIONS/NUM_LEVELS)
    if DIMENSIONS < x*NUM_LEVELS:
        x = x-1
    return (pixbit, d, lgf, c, f, n, adI, adz, zComp, lgCn, logn, x)

def binarizing (n,b):
    s = str(bin(n))
    #print("---" , s , len(s) )
    s = s [2:]
    #print("---" , s )
    while len(s) < b:
        s = '0'+ s
    return s

def inputGen(train_X, mystr, countTimer, NUM_LEVELS, pixbit):
    x = list(binarizing (int(item*NUM_LEVELS),pixbit) for row in train_X for item in row)
    for i in range(len(x)-1):
        if (x[i+1] != x[i]):
            mystr = mystr +' "'+ x[i+1]  +'"'+" after "+ str(countTimer) +" ns,"
        countTimer = countTimer + 1
    return mystr, countTimer

def write_memory(posision, NUM_LEVELS,d):
    weight_mem = []
    for ini in posision:
        strinit2 = ""
        for i in range(len(ini)):
            if ini[i] == -1 :
                strinit2 = strinit2 + '0'
            else :
                strinit2 = strinit2 + '1'
        weight_mem.append(strinit2)
    with open('Xili_HDCMem/BV_img.coe', 'w') as output:
        output.write("memory_initialization_radix=2;\n")
        output.write("memory_initialization_vector=\n")
        for i in weight_mem:
            output.write(i)
            output.write(",\n")
        #output.write(";")
    with open('Xili_HDCMem/BV_img.mif', 'w') as output:
        for i in weight_mem:
            #output.write('"')
            output.write(i)
            #output.write('"')
            output.write(",\n")

    id_mem = []
    poniter =  math.ceil(math.log2(NUM_LEVELS))
    for i in range(2**poniter):
        mystr = ""
        if i == 0 :
            mystr = "0"*d
        elif i == 2**poniter-1 :
            mystr = "1"*d
        else :
            mystr = "0"*(d-(i*c))+"1"*(i*c)
        id_mem.append(mystr)
    #for i in id_mem:
        #print (i)
    with open('Xili_HDCMem/ID_img.coe', 'w') as output:
        output.write("memory_initialization_radix=2;\n")
        output.write("memory_initialization_vector=\n")
        for i in id_mem:
            output.write(i)
            output.write("\n")
        #output.write(";")
    with open('Xili_HDCMem/ID_img.mif', 'w') as output:
        for i in id_mem:
            #output.write('"')
            output.write(i)
            #output.write('"')
            output.write("\n")

def class_normalize_memory (a, mem_size, number_of_confComp, zeropadding):
    for k in range(len(a)):
        for m in range (number_of_confComp):
            mystr =""
            if k == 1:
                flag = 1
            else:
                flag = 0
            if m != number_of_confComp-1:
                for s in range(mem_size):
                    #print((s+((m)*mem_size)))
                    #print(s," -  ", m, " -  ", mem_size, " -  ", (s+((m)*mem_size)))
                    if a[k][(s+((m)*mem_size))] > 0:
                        mystr = '1' + mystr
                    else:
                        mystr = '0' + mystr
            else:
                for s in range(mem_size-zeropadding):
                    #print((s+((m)*mem_size)))
                    if a[k][(s+((m)*mem_size))] > 0:
                        mystr = '1' + mystr
                    else:
                        mystr = '0' + mystr
                zeros = '0'*zeropadding
                mystr = zeros + mystr
            with open('Xili_HDCMem/{}_{}.coe'.format(k, m), 'w') as output:
                output.write("memory_initialization_radix=2;\n")
                output.write("memory_initialization_vector=\n")
                output.write(mystr)
            with open('Xili_HDCMem/{}_{}.txt'.format(k, m), 'w') as output:
                output.write(mystr)
    #for i in a:
    #    for j in a:
    #        print(str(i ^ j).count('1'), end=",	")
    #    print("\n")

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
        self.value = embeddings.Level(levels, out_features)
        print ( "self.position.weight    " , self.position.weight)
        print ( "self.value.weight    " , self.value.weight)
        #print("init : \n" , init_num, "\n", init)
        #print("XORs : \n" , XORs_num, "\n", XORs)
        #print("=============================================================================================")

    def forward(self, x):
        x = self.flatten(x)
        if print_flag == 1 :
            print(" ** self.flatten(x) :")
            print(x)
            print(" ** self.value(x) :")
            print(self.value(x))
        sample_hv = torchhd.bind(self.position.weight, self.value(x))
        if print_flag == 1 :
            print(" ** sample_hv 1 :")
            #print(sample_hv)
        sample_hv = torchhd.multiset(sample_hv)
        if print_flag == 1 :
            print(" ** sample_hv 2 :")
            print(sample_hv)
        positive = torch.tensor(1.0, dtype=sample_hv.dtype, device=sample_hv.device)
        negative = torch.tensor(-1.0, dtype=sample_hv.dtype, device=sample_hv.device)
        if print_flag == 1 :
            print(" ** torch.where(sample_hv > 0, positive, negative) :")
            print(torch.where(sample_hv > 0, positive, negative))
        return torch.where(sample_hv > 0, positive, negative)


encode = Encoder(DIMENSIONS, IMG_SIZE, NUM_LEVELS)
encode = encode.to(device)

num_classes = len(train_ds.classes)
model = Centroid(DIMENSIONS, num_classes)
model = model.to(device)

with torch.no_grad():
    for samples, labels in tqdm(train_ld, desc="Training"):
        samples = samples.to(device)
        labels = labels.to(device)

        samples_hv = encode(samples)
        model.add(samples_hv, labels)

accuracy = torchmetrics.Accuracy("multiclass", num_classes=num_classes)

print(" **** Classs Hyper vectors :")
print(model.weight)
pixbit, d, lgf, c, f, n, adI, adz, zComp, lgCn, logn, x = config (DIMENSIONS, IMG_SIZE*IMG_SIZE, NUM_LEVELS, len(model.weight))
#print(len(model.weight),  type(model.weight), model.weight[1])
class_normalize_memory (model.weight, 2**n, adI, (2**n)*adI - d)
print (pixbit, d, lgf, c, f, n, adI, adz, zComp, lgCn, logn, x )

with torch.no_grad():
    model.normalize()
    print_flag = 1
    count = 0
    mystr = ""
    for samples, labels in tqdm(test_ld, desc="Testing"):
        count = count + 1
        countTimer = countTimer + f + 50
        if count == 5:
            print_flag = 0
        samples = samples.to(device)
        samples_hv = encode(samples)
        outputs = model(samples_hv, dot=True)
        if print_flag == 1:
            #mystr , countTimer = inputGen(samples, mystr, countTimer, NUM_LEVELS, pixbit)
            print(" ----  outputs.cpu(), labels : ", outputs.cpu(), labels)
            print("samples ", samples)
            print("samples_hv ", samples_hv)
            print(" ----  model(samples_hv, dot=True) ", model(samples_hv, dot=True))
        accuracy.update(outputs.cpu(), labels)
#print(a , "\n", b)
print(f"Testing accuracy of {(accuracy.compute().item() * 100):.3f}%")
#print(mystr)
