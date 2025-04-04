import torch
import numpy as np
import os
import math

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
            with open(path+'mem/normal/{}_{}.mif'.format(k, number_of_confComp-m-1), 'w') as output:
                output.write(mystr[mem_size*(m):mem_size*(m+1)])

def group_of_chvs(mem_size, number_of_confComp, zeropadding, path):
    chvs = torch.load(path+"/model/chvs.pt")
    bin_list = []
    for k in chvs:
        binOfEach = ""
        for s in k:
                if s > 0:
                        binOfEach = binOfEach+'1'
                else:
                        binOfEach = binOfEach+'0'
        zeros = '0'*zeropadding
        binOfEach = zeros + binOfEach
        bin_list.append(binOfEach)

    # Write all CHV memory in one file
    file_path = os.path.join(path+f'mem/normal/CHV_img.mif')
    with open(file_path, "w") as f:
        for classes in bin_list:
            f.write(classes + "\n")  # Write each segment on a new line



def class_normalize_memory_sparse(ls, mem_size, number_of_confComp, zeropadding, path):
    a = torch.load(path+"/model/chvs.pt")
    for k in range(len(a)):
        mystr =""
        indices_to_keep = [i not in ls for i in np.arange(0,len(a[k]))]
        mystr = "".join(["1" if a_i > 0 else "0" for a_i in a[k][indices_to_keep]])
        
        zeros = '0'*zeropadding
        mystr = zeros + mystr
        for m in range(number_of_confComp):
            with open(path+'mem/sparse/{}_{}.mif'.format(k, number_of_confComp-m-1), 'w') as output:
                output.write(mystr[mem_size*(m):mem_size*(m+1)])

def write_memory(path, dimensions, levels, lfsr=True):
    if lfsr:
        position = np.array(torch.load(path+"model/sequence.pt")).astype(np.int8)
    else:
        position = torch.load(path+"model/BV.pt").detach().numpy().astype(np.int8)
    #memory specific to LFSR encoding
    if lfsr:
        XORs     = torch.load(path+"model/xors.pt")
        strXors = ""
        for i in range(dimensions):
            if i in XORs :
                strXors = strXors + '1'
            else:
                strXors = strXors + '0'
        with open(path+'mem/configSignature.txt', 'w') as output:
            output.write(str(strXors[::-1]))    
    
    #general memory file (sequence in LFSR is position in BV)
    weight_mem = []
    for ini in position:
        # test = ""
        # for i in range(len(ini)):
        #     if ini[i] == -1 :
        #         test = test + '0'
        #     else :
        #         test = test + '1'
        # scale [-1,1] to [0,1] by doing (x+1) // 2
        ini = (ini+1) // 2
        strinit2 = "".join([str(i) for i in ini])
        # assert test == strinit2
        weight_mem.append(strinit2)

    with open(path+'mem/configInitialvalues.txt', 'w') as output:
        output.write(str(weight_mem[0]))
            
    with open(path+'mem/BV_img.coe', 'w') as output:
        output.write("memory_initialization_radix=2;\n")
        output.write("memory_initialization_vector=\n")
        for i in weight_mem:
            output.write(i)
            output.write(",\n")
            
    with open(path+'mem/BV_img.mif', 'w') as output:
        for i in weight_mem:
            output.write(i)
            output.write(",\n")

    c = int(math.floor(dimensions/levels))
    id_mem = []
    if lfsr:
        pointer =  math.ceil(math.log2(levels))
        for i in range(2**pointer):
            mystr = ""
            if i == 0 :
                mystr = "0"*dimensions
            elif i == 2**pointer-1 :
                mystr = "1"*dimensions
            else :
                mystr = "0"*(dimensions-(i*c))+"1"*(i*c)
            id_mem.append(mystr)
    else:
        #same as writing BV
        ids = torch.load(path+"model/ID.pt").detach().numpy().astype(np.int8)
        for id in ids:
            # test = ""
            # for i in range(len(ini)):
            #     if ini[i] == -1 :
            #         test = test + '0'
            #     else :
            #         test = test + '1'
            id = (id+1) // 2
            strinit2 = "".join([str(i) for i in id])
            # assert test == strinit2
            id_mem.append(strinit2)
        
    with open(path+'/mem/ID_img.coe', 'w') as output:
        output.write("memory_initialization_radix=2;\n")
        output.write("memory_initialization_vector=\n")
        for i in id_mem:
            output.write(i)
            output.write("\n")
        output.write(";")

def gen_sparsemodule(path, ls, DIMENSIONS):
    os.system('touch '+path+'/connector.vhd')
    f = open(path+'/connector.vhd', "w")
    f.write("LIBRARY IEEE; \nUSE IEEE.STD_LOGIC_1164.ALL; \nUSE IEEE.NUMERIC_STD.ALL; \n  \nENTITY connector IS \n\tGENERIC(d : INTEGER := 1000; ----dimentionsize \n\tp: INTEGER:= 1000 ); --- prunsize \n\tPORT ( \n\t\tinput         : IN  STD_LOGIC_VECTOR (d-1 DOWNTO 0); \n\t\tpruneoutput        : OUT  STD_LOGIC_VECTOR (p-1 DOWNTO 0)      \n\t);\nEND ENTITY connector;\n\nARCHITECTURE behavioral OF connector  IS\nBEGIN\n")
    counter = 0 
    for i in range(DIMENSIONS):
        if i not in ls:
            f.write("\t pruneoutput("+str(DIMENSIONS- len(ls) - counter-1)+") <= input("+str(DIMENSIONS-i-1)+");\n")
            counter = counter + 1
    f.write('\nEND ARCHITECTURE behavioral;')
    f.close()
    return (len(ls))
