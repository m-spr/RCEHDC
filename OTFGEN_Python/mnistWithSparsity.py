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
os.system('mkdir MNISTmodels')

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
    for j in range(len(a[0])):
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
    x = math.ceil(DIMENSIONS/NUM_LEVELS) ## why not floor? 
    if DIMENSIONS < x*NUM_LEVELS:
        x = x-1
    return (pixbit, d, d-sparse, lgf, c, f, n, adI, adz, zComp, lgCn, logn, r,  x)

  
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

def write_memory(XORs, init_num, posision, NUM_LEVELS,d, value):
    print("write_memory")
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
    value_mem = []
    for ini in value:
        strinit2 = ""
        for i in range(len(ini)):
            if ini[i] == -1 :
                strinit2 = strinit2 + '0'
            else :
                strinit2 = strinit2 + '1'
        value_mem.append(strinit2)
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
    

def BV_ID_memory_sparse (posision, ls):
    weight_mem = []
    with open('mem/BV_img_sparse.mif', 'w') as output:
        print(len(posision))
        for ini in posision:
            indices_to_keep = [i not in ls for i in np.arange(0,len(ini))]
            mystr = "".join(["1" if a_i > 0 else "0" for a_i in ini[indices_to_keep]])  # for c in range(len(a))]
            output.write(mystr)
            output.write("\n")


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
    print("class_normalize_memory")
    for k in range(len(a)):
        print(st(a[k]))
        mystr =""
        #print(a[k])
        for m in a[k]:
            
            if m > 0 :
                mystr =  mystr + "1" 
            else:
                mystr =  mystr + "0" 
        zeros = '0'*zeropadding
        mystr = zeros + mystr
        #print("     ",k ,"  : ",mystr)
        for m in range(number_of_confComp):
            #print(mem_size*(m+1)-1,mem_size*(m), mystr[mem_size*(m):mem_size*(m+1)])

            with open('../OTFGEN_VHDL/normalHDC/{}_{}.mif'.format(k, number_of_confComp-m-1), 'w') as output:
                output.write(mystr[mem_size*(m):mem_size*(m+1)])

    #for i in a:
    #    for j in a:
    #        print(str(i ^ j).count('1'), end=",	")
    #    print("\n")
def Sparsemodule(ls):
    #os.system('mkdir MNISTmodels/sparseFiles')
    os.system('touch ../OTFGEN_VHDL/SparseHDC/connector.vhd')
    f = open('../OTFGEN_VHDL/SparseHDC/connector.vhd', "w")
    f.write("LIBRARY IEEE; \nUSE IEEE.STD_LOGIC_1164.ALL; \nUSE IEEE.NUMERIC_STD.ALL; \n  \nENTITY connector IS \n\tGENERIC(d : INTEGER := 1000; ----dimentionsize \n\tp: INTEGER:= 1000 ); --- prunsize \n\tPORT ( \n\t\tinput         : IN  STD_LOGIC_VECTOR (d-1 DOWNTO 0); \n\t\tpruneoutput        : OUT  STD_LOGIC_VECTOR (p-1 DOWNTO 0)      \n\t);\nEND ENTITY connector;\n\nARCHITECTURE behavioral OF connector  IS\nBEGIN\n")
    counter = 0 
    for i in range(DIMENSIONS):
        if i in ls:
            pass
        else:
            f.write("\t pruneoutput("+str(DIMENSIONS- len(ls) - counter-1)+") <= input("+str(DIMENSIONS-i-1)+");\n")
            counter = counter + 1
    f.write('\nEND ARCHITECTURE behavioral;')
    f.close()

# def class_normalize_memory_sparse (a, mem_size, number_of_confComp, zeropadding, ls):
#     for k in range(len(a)):
#         mystr =""
#         for m in range(len(a[k])):
#             if m in ls:
#                 pass
#             else:
#                 if a[k][m] > 0 :
#                     mystr =  "1" + mystr
#                 else:
#                     mystr =  "0" + mystr 
#         zeros = '0'*zeropadding
#         mystr = zeros + mystr
#         print(mystr)
#         for m in range(number_of_confComp):
#             with open('../OTFGEN_VHDL/SparseHDC/{}_{}.mif'.format(k, number_of_confComp-m-1), 'w') as output:
#                 output.write(mystr[mem_size*(m):mem_size*(m+1)])

def class_normalize_memory_sparse (a, mem_size, number_of_confComp, zeropadding, ls):
    for k in range(len(a)):
        mystr =""
        indices_to_keep = [i not in ls for i in np.arange(0,DIMENSIONS)]
        mystr = ["".join(["1" if a_i > 0 else "0" for a_i in a[k][indices_to_keep]]) for k in num_classes]
        # for m in range(len(a[k])):
        #     if m in ls:
        #         pass
        #     else:
        #         if a[k][m] > 0 :
        #             mystr =  mystr + "1" 
        #         else:
        #             mystr =  mystr + "0" 
        indices_to_keep = [i not in ls for i in np.arange(0,len(a[k]))]
        mystr = "".join(["1" if a_i > 0 else "0" for a_i in a[k][indices_to_keep]])  # for c in range(len(a))]
        # for m in range(len(a[k])):
        #     if m in ls:
        #         pass
        #     else:
        #         if a[k][m] > 0 :
        #             mystr =  mystr + "1" 
        #         else:
        #             mystr =  mystr + "0" 
        zeros = '0'*zeropadding
        mystr = zeros + mystr
        print(mystr)
        for m in range(number_of_confComp):
            with open('../OTFGEN_VHDL/SparseHDC/{}_{}.mif'.format(k, number_of_confComp-m-1), 'w') as output:
                output.write(mystr[mem_size*(m):mem_size*(m+1)])


    # for k in range(len(a)):
    #     for m in range (number_of_confComp):
    #         mystr =""
    #         if k == 1:
    #             flag = 1
    #         else:
    #             flag = 0
    #         if m != number_of_confComp-1:
    #             for s in range(mem_size):
    #                 if (s+((m)*mem_size)) in ls:
    #                     pass
    #                     #print(s+((m)*mem_size), end = ", ")
    #                 #print((s+((m)*mem_size)))
    #                 #print(s," -  ", m, " -  ", mem_size, " -  ", (s+((m)*mem_size)))
    #                 else:
    #                     if a[k][(s+((m)*mem_size))] > 0:
    #                         mystr = '1' + mystr
    #                     else:
    #                         mystr = '0' + mystr
    #         else:
    #             for s in range(mem_size-zeropadding):
    #                 if (s+((m)*mem_size)) in ls:
    #                     pass
    #                     #print(s+((m)*mem_size), end = ", ")
    #                 else:
    #                     if a[k][(s+((m)*mem_size))] > 0:
    #                         mystr = '1' + mystr
    #                     else:
    #                         mystr = '0' + mystr
    #             zeros = '0'*zeropadding
    #             mystr = zeros + mystr
    #             ----------------------------------------------------------------------
    #         with open('./mem/sparseFiles/sparse{}_{}.coe'.format(k, m), 'w') as output:
    #             output.write("memory_initialization_radix=2;\n")
    #             output.write("memory_initialization_vector=\n")
    #             output.write(mystr)
    #         with open('../OTFGEN_VHDL/SparseHDC/sparse{}_{}.mif'.format(k, m), 'w') as output:
    #             output.write(mystr)

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
        #ls = [0, 1, 2, 3, 4, 5, 6, 9, 10, 11, 12, 13, 14, 16, 17, 21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 37, 38, 39, 42, 43, 44, 49, 50, 51, 52, 53, 54, 55, 59, 60, 61, 64, 65, 66, 67, 68, 69, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 107, 108, 109, 110, 111, 112, 118, 121, 123, 124, 125, 126, 127, 128, 129, 132, 133, 135, 136, 138, 139, 140, 141, 146, 150, 155, 156, 157, 158, 159, 161, 163, 164, 165, 166, 170, 171, 172, 173, 176, 177, 178, 183, 184, 185, 187, 188, 190, 191, 193, 194, 195, 196, 200, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 260, 261, 262, 263, 264, 270, 271, 272, 273, 274, 275, 276, 277, 279, 280, 281, 282, 283, 284, 285, 288, 289, 290, 291, 292, 293, 297, 299, 300, 301, 302, 307, 308, 309, 310, 311, 312, 313, 314, 315, 316, 317, 318, 319, 320, 321, 323, 324, 325, 326, 327, 328, 329, 332, 333, 334, 335, 336, 337, 340, 341, 346, 347, 348, 349, 353, 354, 355, 356, 367, 368, 369, 370, 374, 375, 376, 381, 383, 384, 385, 386, 387, 388, 389, 390, 392, 393, 394, 395, 401, 406, 407, 408, 409, 415, 418, 419, 420, 421, 422, 423, 424, 427, 428, 429, 430, 432, 433, 434, 435, 436, 437, 438, 439, 440, 441, 442, 443, 444, 451, 452, 453, 454, 455, 456, 457, 458, 459, 460, 461, 462, 463, 464, 465, 466, 467, 475, 482, 485, 486, 487, 488, 489, 490, 491, 492, 493, 494, 495, 496, 497, 498, 499, 500, 501, 502, 503, 504, 505, 506, 507, 508, 509, 510, 511, 512, 513, 514, 515, 516, 517, 518, 519, 520, 521, 524, 525, 526, 527, 528, 529, 531, 532, 533, 534, 535, 536, 538, 539, 540, 541, 542, 543, 545, 546, 547, 548, 549, 550, 551, 552, 553, 555, 556, 557, 558, 559, 560, 563, 565, 566, 567, 568, 569, 572, 575, 576, 578, 579, 580, 581, 582, 583, 584, 586, 587, 588, 589, 591, 593, 594, 597, 598, 599, 600, 601, 602, 604, 605, 606, 607, 608, 612, 613, 614, 615, 616, 617, 623, 624, 625, 630, 631, 633, 634, 635, 636, 637, 638, 642, 643, 644, 645, 646, 647, 649, 653, 654, 655, 656, 657, 658, 659, 660, 661, 662, 663, 664, 665, 670, 671, 672, 673, 674, 675, 676, 677, 679, 680, 681, 682, 687, 688, 689, 690, 691, 692, 693, 694, 695, 698, 699, 702, 703, 704, 709, 710, 711, 712, 713, 714, 718, 719, 720, 721, 722, 723, 724, 725, 726, 729, 730, 731, 734, 742, 744, 745, 746, 747, 748, 751, 752, 753, 754, 755, 756, 757, 758, 759, 760, 761, 762, 763, 764, 765, 775, 776, 779, 780, 781, 782, 783, 784, 785, 786, 787, 789, 790, 791, 792, 796, 798, 799, 802, 803, 804, 805, 806, 807, 810, 811, 813, 816, 817, 821, 822, 826, 827, 828, 829, 830, 831, 832, 833, 834, 835, 837, 838, 839, 840, 841, 842, 843, 844, 845, 846, 847, 851, 852, 853, 854, 855, 856, 857, 859, 860, 867, 868, 869, 870, 871, 877, 878, 879, 880, 881, 882, 883, 884, 885, 891, 892, 893, 894, 896, 897, 898, 900, 901, 907, 908, 909, 910, 911, 912, 914, 915, 917, 918, 919, 920, 924, 925, 926, 927, 928, 934, 935, 940, 941, 942, 943, 944, 945, 946, 947, 948, 949, 950, 951, 954, 955, 956, 957, 958, 960, 961, 962, 965, 966, 967, 968, 969, 972, 975, 979, 980, 983, 987, 990, 992, 993, 994, 997, 998, 999]
        #BV_ID_memory_sparse (arr, ls)
        #genConfig (XORs, init_num, DIMENSIONS)
        write_memory(XORs, init_num, generated_sequence, NUM_LEVELS,DIMENSIONS, self.value.weight)

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
    for samples, labels in test_ld:
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
#print("0 for 0 ", model.weight[0])

ls = class_sparsity (model.weight)
#BV_ID_memory_sparse (model.position,  ls)
print(ls)
print ("sparse model:", sparseconfig (DIMENSIONS, len(ls), IMG_SIZE*IMG_SIZE, NUM_LEVELS, len(model.weight)))
print ("Normal model:  \npixbit, d, lgf, c, f, n, adI, adz, zComp, lgCn, logn, x \n", config (DIMENSIONS, IMG_SIZE*IMG_SIZE, NUM_LEVELS, len(model.weight)))
pixbit, d, lgf, c, f, n, adI, adz, zComp, lgCn, logn,r, x  = config (DIMENSIONS, IMG_SIZE*IMG_SIZE, NUM_LEVELS, len(model.weight))
Sparsemodule(ls)
class_normalize_memory (model.weight, 2**n, adI, (2**n)*adI - d)

pixbit, d, sparse, lgf, c, f, n, adI, adz, zComp, lgCn, logn,r, x = sparseconfig (DIMENSIONS, len(ls), IMG_SIZE*IMG_SIZE, NUM_LEVELS, len(model.weight))

class_normalize_memory_sparse (model.weight, 2**n, adI, (2**n)*adI- (sparse), ls)
"""
print ("Normal model: ", pixbit, d, lgf, c, f, n, adI, adz, zComp, lgCn, logn, x , DIMENSIONS-spa )
print ("sparse model: ", DIMENSIONS, config (DIMENSIONS-spa, IMG_SIZE*IMG_SIZE, NUM_LEVELS, len(model.weight)))

#print(config (3000, IMG_SIZE*IMG_SIZE, NUM_LEVELS, len(model.weight)))
class_normalize_memory (model.weight, 2**n, adI, (2**n)*adI - d)
"""    