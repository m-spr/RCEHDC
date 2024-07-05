import torch
import numpy as np

def class_sparsity (a):     # a is model.weight
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

def class_normalize_memory(mem_size, number_of_confComp, zeropadding, path):
    a = torch.load(path+"/model/chvs.pt")
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
        
        for m in range(number_of_confComp):
            with open(path+'mem/normal/full{}_{}.mif'.format(k, number_of_confComp-m-1), 'w') as output:
                output.write(mystr[mem_size*(m):mem_size*(m+1)])

def class_normalize_memory_sparse (a, mem_size, number_of_confComp, zeropadding, ls):
    for k in range(len(a)):
        mystr =""
        indices_to_keep = [i not in ls for i in np.arange(0,len(a[k]))]
        mystr = "".join(["1" if a_i > 0 else "0" for a_i in a[k][indices_to_keep]])# for c in range(len(a))]
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
        for m in range(number_of_confComp):
            with open('../OTFGEN_VHDL/SparseHDC/{}_{}.mif'.format(k, number_of_confComp-m-1), 'w') as output:
                output.write(mystr[mem_size*(m):mem_size*(m+1)])

def write_memory(path, DIMENSIONS):
    XORs     = torch.load(path+"model/xors.pt")
    #init_num = torch.load(path+"/model/init_num.pt")
    position = torch.load(path+"model/sequence.pt")

    strXors = ""
    for i in range(DIMENSIONS):
        if i in XORs :
              strXors = strXors + '1'
        else:
              strXors = strXors + '0'
    with open(path+'mem/configSignature.txt', 'w') as output:
        output.write(str(strXors[::-1]))
    weight_mem = []
    for ini in position:
        strinit2 = ""
        for i in range(len(ini)):
            if ini[i] == -1 :
                strinit2 = strinit2 + '0'
            else :
                strinit2 = strinit2 + '1'
        weight_mem.append(strinit2)
    #strinit = weight_mem[0]
    with open(path+'mem/configInitialvalues.txt', 'w') as output:
        output.write(str(weight_mem[0]))
    with open(path+'mem/BV_img.coe', 'w') as output:
        output.write("memory_initialization_radix=2;\n")
        output.write("memory_initialization_vector=\n")
        for i in weight_mem:
            output.write(i)
            output.write(",\n")
        #output.write(";")
    with open(path+'mem/BV_img.mif', 'w') as output:
        for i in weight_mem:
            #output.write('"')
            output.write(i)
            #output.write('"')
            output.write(",\n")