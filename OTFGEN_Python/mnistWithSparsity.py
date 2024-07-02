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
np.set_printoptions(threshold=sys.maxsize)
torch.set_printoptions(threshold=sys.maxsize)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using {} device".format(device))

DIMENSIONS = 1100
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

train_ds = MNIST("./data", train=True, transform=transform, download=True)   #, transform=transform
train_ld = torch.utils.data.DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

test_ds = MNIST("./data", train=False, transform=transform, download=True) #, transform=transform
test_ld = torch.utils.data.DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)


os.system('mkdir mem')
os.system('mkdir mem/Xili_HDCMem')
os.system('mkdir mem/sparseFiles')

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


def class_sparsity (a):     # a is model.weight
    #class1 = torch.load( 'mnist_LFSREncoder_quantize_True_ret_1000.pt' , map_location=torch.device('cpu'))
    count = 0
    ls = []
    for j in range(len(a[1])):
        m = a[0][j]
        n = 0
        for i in range(len(a)):
            if m != a[i][j]:
                n = 1
        if n == 0:
            count = count + 1
            ls.append(j)
    print("Number of pruning: ",count)
    return(ls)
def st(hv):
    str = ""
    for i in hv:
        # print(int(i))
        if int(i) > 0:
            str = str + '1'
        else:
            str = str + '0'
        #str = str + " "
    #print(str) 
def sparseconfig (DIMENSIONS, sparse, featureSize, NUM_LEVELS, classes):
    pixbit = math.ceil(math.log2(NUM_LEVELS))
    d = DIMENSIONS
    lgf  = math.ceil(math.log2(featureSize))
    c   = classes
    f = featureSize
    n , adI = memMinimizer (DIMENSIONS - sparse, featureSize)
    adz = 2**(math.ceil(math.log2(adI)))-adI
    zComp =  2**(math.ceil(math.log2(classes)))-classes
    lgCn =  math.ceil(math.log2(classes))
    logn = math.ceil(math.log2(adI))
    x = math.ceil(DIMENSIONS/NUM_LEVELS)
    if DIMENSIONS < x*NUM_LEVELS:
        x = x-1
    return (pixbit, d, sparse, lgf, c, f, n, adI, adz, zComp, lgCn, logn, r,  x)

  
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
    return (pixbit, d, lgf, c, f, n, adI, adz, zComp, lgCn, logn, r, x)

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

def genConfig (XORs, init_num, DIMENSIONS):
    strXors = ""
    strinit = ""
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
    with open('mem/lfsrConfig.txt', 'w') as output:
        output.write("XORs\n")
        output.write(str(strXors[::-1]))
        output.write("\n")
        output.write("init\n")
        output.write(strinit[::-1])

def write_memory(XORs, init_num, posision, NUM_LEVELS,d):
    strXors = ""
    for i in range(DIMENSIONS):
        if i in XORs :
              strXors = strXors + '1'
        else:
              strXors = strXors + '0'
    with open('mem/configSigniture.mif', 'w') as output:
        output.write(str(strXors[::-1]))
    weight_mem = []
    for ini in posision:
        strinit2 = ""
        for i in range(len(ini)):
            if ini[i] == -1 :
                strinit2 = strinit2 + '0'
            else :
                strinit2 = strinit2 + '1'
        weight_mem.append(strinit2)
    #strinit = weight_mem[0]
    with open('mem/configInitialvalues.mif', 'w') as output:
        output.write(str(weight_mem[0]))
    with open('mem/BV_img.coe', 'w') as output:
        output.write("memory_initialization_radix=2;\n")
        output.write("memory_initialization_vector=\n")
        for i in weight_mem:
            output.write(i)
            output.write(",\n")
        #output.write(";")
    with open('mem/BV_img.mif', 'w') as output:
        for i in weight_mem:
            #output.write('"')
            output.write(i)
            #output.write('"')
            output.write(",\n")
    ########## old version 
    # id_mem = []
    # poniter =  math.ceil(math.log2(NUM_LEVELS))
    # for i in range(2**poniter):
    #     mystr = ""
    #     if i == 0 :
    #         mystr = "0"*d
    #     elif i == 2**poniter-1 :
    #         mystr = "1"*d
    #     else :
    #         mystr = "0"*(d-(i*c))+"1"*(i*c)
    #     id_mem.append(mystr)
    # #for i in id_mem:
        #print (i)
    # with open('mem/ID_img.coe', 'w') as output:
    #     output.write("memory_initialization_radix=2;\n")
    #     output.write("memory_initialization_vector=\n")
    #     for i in id_mem:
    #         output.write(i)
    #         output.write("\n")
    #     #output.write(";")
    # with open('mem/ID_img.mif', 'w') as output:
    #     for i in id_mem:
    #         #output.write('"')
    #         output.write(i)
    #         #output.write('"')
    #         output.write("\n")

def class_normalize_memory (a, mem_size, number_of_confComp, zeropadding):
    for k in range(len(a)):
        mystr =""
        #print(a[k])
        for m in a[k]:
            
            if m > 0 :
                mystr = mystr +  '1'
            else:
                mystr = mystr +  '0'
        zeros = '0'*zeropadding
        mystr = zeros + mystr
        #print(mystr)
        for m in range(number_of_confComp):
            #print(mem_size*(m+1)-1,mem_size*(m), mystr[mem_size*(m):mem_size*(m+1)])
            with open('../OTFGEN_VHDL/normalHDC/full{}_{}.mif'.format(k, number_of_confComp-m-1), 'w') as output:
                output.write(mystr[mem_size*(m):mem_size*(m+1)])

def class_normalize_memory2 (a, mem_size, number_of_confComp, zeropadding):
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
                        mystr = mystr +  '1'
                    else:
                        mystr = mystr + '0'
            else:
                for s in range(mem_size-zeropadding):
                    #print((s+((m)*mem_size)))
                    if a[k][(s+((m)*mem_size))] > 0:
                        mystr = mystr +  '1'
                    else:
                        mystr = mystr + '0'
                zeros = '0'*zeropadding
                mystr = zeros + mystr
            # number_of_confComp-m OR m
            with open('mem/Xili_HDCMem/full{}_{}.coe'.format(k, m), 'w') as output:
                output.write("memory_initialization_radix=2;\n")
                output.write("memory_initialization_vector=\n")
                output.write(mystr)
            with open('../OTFGEN_VHDL/normalHDC/full{}_{}.mif'.format(k, m), 'w') as output:
                output.write(mystr)
    #for i in a:
    #    for j in a:
    #        print(str(i ^ j).count('1'), end=",	")
    #    print("\n")
def Sparsemodule(ls):
    #os.system('mkdir MNISTmodels/sparseFiles')
    os.system('touch ../OTFGEN_VHDL/SparseHDC/connector.vhd')
    f = open('../OTFGEN_VHDL/SparseHDC/connector.vhd', "w")
    f.write("LIBRARY IEEE; \nUSE IEEE.STD_LOGIC_1164.ALL; \nUSE IEEE.NUMERIC_STD.ALL; \n  \nENTITY connector IS \n\tPORT ( \n\t\tinput         : IN  STD_LOGIC_VECTOR ("+ str(d - 1)+" DOWNTO 0); \n\t\tpruneoutput        : OUT  STD_LOGIC_VECTOR ("+ str(len(ls)- 1)+" DOWNTO 0)      \n\t);\nEND ENTITY connector;\n\nARCHITECTURE behavioral OF connector  IS\nBEGIN\n")
    counter = 0 
    for i in range(1000):
        if i in ls:
            f.write("pruneoutput("+str(counter)+") <= input("+str(i)+");\n")
            counter = counter + 1
    f.write('\nEND ARCHITECTURE behavioral;')
    f.close()

def class_normalize_memory_sparse (a, mem_size, number_of_confComp, zeropadding, ls):
    for k in range(len(a)):
        for m in range (number_of_confComp):
            mystr =""
            if k == 1:
                flag = 1
            else:
                flag = 0
            if m != number_of_confComp-1:
                for s in range(mem_size):
                    if (s+((m)*mem_size)) in ls:
                        pass
                        #print(s+((m)*mem_size), end = ", ")
                    #print((s+((m)*mem_size)))
                    #print(s," -  ", m, " -  ", mem_size, " -  ", (s+((m)*mem_size)))
                    else:
                        if a[k][(s+((m)*mem_size))] > 0:
                            mystr = '1' + mystr
                        else:
                            mystr = '0' + mystr
            else:
                for s in range(mem_size-zeropadding):
                    if (s+((m)*mem_size)) in ls:
                        pass
                        #print(s+((m)*mem_size), end = ", ")
                    else:
                        if a[k][(s+((m)*mem_size))] > 0:
                            mystr = '1' + mystr
                        else:
                            mystr = '0' + mystr
                zeros = '0'*zeropadding
                mystr = zeros + mystr
            with open('./mem/sparseFiles/sparse{}_{}.coe'.format(k, m), 'w') as output:
                output.write("memory_initialization_radix=2;\n")
                output.write("memory_initialization_vector=\n")
                output.write(mystr)
            with open('../OTFGEN_VHDL/SparseHDC/sparse{}_{}.mif'.format(k, m), 'w') as output:
                output.write(mystr)

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
        np.save("./mem/mnist_levels.npy", arr)
        tArr = torch.nn.Parameter(torchhd.MAPTensor(torch.from_numpy(arr).float()))
        self.value.weight = tArr.float()

        init_num = random.randint(1, 2**out_features)
        #init_num = 4005700534675144948017520162097884868995542063410121040524186571954440390130004122707793153004062360041636382462054526905859450700091502234942087738732084346636139513840569559345488256204180986892293042541798471825399139750542421374166982260754066786698152028283234760575400288349132987005389102090211
        XORs_num = random.randint(2**(out_features-1), 2**out_features)
        #XORs_num = 8304480432096973745855277495005820904147225062766752395232516640190584805313156405292672642268352808496341593728537662541323806162655398579460968638548439233518083677818779822153866372461399193020495636799816940385452684276171359521470779611083416341728702669855951160018696384825076986035152844977163        print(init_num)
        #print(XORs_num)
        init = [eval(i) for i in [*bin(init_num)[2:]]]
        init.extend([0] * (out_features - len(init)))
        XORs = [i for i, x in enumerate(reversed([*bin(XORs_num)[2:]])) if x == '1']
        lfsr = LFSR(init, XORs)
        sequence_length = size * size
        generated_sequence = lfsr.generate_sequence(sequence_length)
        arr = np.array(generated_sequence)
        #print(arr)
        #np.save("./mem/mnist_bv.npy", arr)
        tArr = torch.nn.Parameter(torchhd.MAPTensor(torch.from_numpy(arr).float()))
        self.position.weight = tArr.float()
    
        #genConfig (XORs, init_num, DIMENSIONS)
        write_memory(XORs, init_num, generated_sequence, NUM_LEVELS,DIMENSIONS)

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


with torch.no_grad():
    for samples, labels in tqdm(train_ld, desc="Training"):
        samples = samples.to(device)
        labels = labels.to(device)

        samples_hv = encode(samples)
        model.add(samples_hv, labels)

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

torch.save(model.weight, "MNISTmodels/mnist.pt")
torch.save(encode, "MNISTmodels/enc_mnist.pt")
#print(encode)
#print(st(model.weight[0]))

ls = class_sparsity (model.weight)

print ("Normal model: ", config (DIMENSIONS, IMG_SIZE*IMG_SIZE, NUM_LEVELS, len(model.weight)))
print ("sparse model: ", sparseconfig (DIMENSIONS, len(ls), IMG_SIZE*IMG_SIZE, NUM_LEVELS, len(model.weight)))
pixbit, d, lgf, c, f, n, adI, adz, zComp, lgCn, logn,r, x  = config (DIMENSIONS, IMG_SIZE*IMG_SIZE, NUM_LEVELS, len(model.weight))
Sparsemodule(ls)
class_normalize_memory (model.weight, 2**n, adI, (2**n)*adI - d)

pixbit, d, sparse, lgf, c, f, n, adI, adz, zComp, lgCn, logn,r, x = sparseconfig (DIMENSIONS, len(ls), IMG_SIZE*IMG_SIZE, NUM_LEVELS, len(model.weight))
class_normalize_memory_sparse (model.weight, 2**n, adI, (2**n)*adI - d, ls)

"""
print ("Normal model: ", pixbit, d, lgf, c, f, n, adI, adz, zComp, lgCn, logn, x , DIMENSIONS-spa )
print ("sparse model: ", DIMENSIONS, config (DIMENSIONS-spa, IMG_SIZE*IMG_SIZE, NUM_LEVELS, len(model.weight)))

#print(config (3000, IMG_SIZE*IMG_SIZE, NUM_LEVELS, len(model.weight)))
class_normalize_memory (model.weight, 2**n, adI, (2**n)*adI - d)
"""    