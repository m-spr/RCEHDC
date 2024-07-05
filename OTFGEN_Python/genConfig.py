import sympy as sp
from sympy import Eq, Symbol, solve, N
import math

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


def sparseconfig (DIMENSIONS, sparse, featureSize, NUM_LEVELS, classes):
    r = int (DIMENSIONS%NUM_LEVELS)
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
    
    config = {
        "in_width"      : pixbit,
        "dim_size"      : d,
        "sparsity"      : sparse,
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

  
def config (DIMENSIONS, featureSize, NUM_LEVELS, classes):
    r = int (DIMENSIONS%NUM_LEVELS)
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
    config = {
        "in_width"      : pixbit,
        "dim_size"      : d,
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

#pixbit, d, lgf, c, f, n, adI, adz, zComp, lgCn, logn, x = config (1000, 28*28, 1000, 9)
#print(pixbit, d, lgf, c, f, n, adI, adz, zComp, lgCn, logn, x)