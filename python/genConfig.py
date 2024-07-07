import math
import torch

def class_sparsity(path):
    a = torch.load(path+"/model/chvs.pt")
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
    print("Dimension size after pruning: ",len(a[0])-count)
    return(ls)

def memMinimizer(d, f):
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


def sparseconfig(dimensions, features, sparse, num_levels, classes):
    r       = int(dimensions%num_levels)
    pixbit  = math.ceil(math.log2(num_levels))
    lgf     = math.ceil(math.log2(features))
    c       = classes
    f       = features
    n , adI = memMinimizer(dimensions - sparse, features)
    adz     = 2**(math.ceil(math.log2(adI)))-adI
    zComp   = 2**(math.ceil(math.log2(classes)))-classes
    lgCn    = math.ceil(math.log2(classes))
    logn    = math.ceil(math.log2(adI))
    x       = math.ceil(dimensions/num_levels)

    if dimensions < x*num_levels:
        x = x-1
    
    config = {
        "in_width"      : pixbit,
        "dim_size"      : dimensions,
        "sparsity"      : dimensions-sparse,
        "lgf"           : lgf,
        "num_classes"   : c,
        "feature_size"  : f,
        "n"             : n,
        "adI"           : adI,
        "adz"           : adz,
        "zComp"         : zComp,
        "lgCn"          : lgCn,
        "logn"          : logn,
        "remainder"     : r,
        "x"             : x
    }
    return config

  
def config(dimensions, features, num_levels, classes):
    r       = int(dimensions%num_levels)
    pixbit  = math.ceil(math.log2(num_levels))
    lgf     = math.ceil(math.log2(features))
    c       = classes
    f       = features
    n , adI = memMinimizer(dimensions, features)
    adz     = 2**(math.ceil(math.log2(adI)))-adI
    zComp   = 2**(math.ceil(math.log2(classes)))-classes
    lgCn    = math.ceil(math.log2(classes))
    logn    = math.ceil(math.log2(adI))
    x       = math.ceil(dimensions/num_levels)

    if dimensions < x*num_levels:
        x = x-1
        
    config = {
        "in_width"      : pixbit,
        "dim_size"      : dimensions,
        "sparsity"      : 0,
        "lgf"           : lgf,
        "num_classes"   : c,
        "feature_size"  : f,
        "n"             : n,
        "adI"           : adI,
        "adz"           : adz,
        "zComp"         : zComp,
        "lgCn"          : lgCn,
        "logn"          : logn,
        "remainder"     : r,
        "x"             : x
    }
    return config