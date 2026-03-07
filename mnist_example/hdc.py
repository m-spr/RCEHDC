import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.datasets import MNIST

# Note: this example requires the torchmetrics library: https://torchmetrics.readthedocs.io
import torchmetrics
from tqdm import tqdm

import torchhd
from quant_models import Centroid
from torchhd import embeddings

import math

import numpy as np
import random
import sys
import os
import json

import pathlib
path = str(pathlib.Path(__file__).parent.resolve())

np.set_printoptions(threshold=sys.maxsize)
torch.set_printoptions(threshold=sys.maxsize)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using {} device".format(device))

with open(path+'/config.json') as f:
    d = json.load(f)
    print(d)

DIMENSIONS = d["DIMENSIONS"]
IMG_SIZE = 28
NUM_LEVELS = d["NUM_LEVELS"]
c = int(math.floor(DIMENSIONS/NUM_LEVELS))
r = int (DIMENSIONS%NUM_LEVELS)
print_flag = 0 
countTimer = 100
BATCH_SIZE = 32
ENCODE_BATCH_SIZE = 32  # batch size for pre-encoding in online learning
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

class LFSREncoder(nn.Module):
    def __init__(self, out_features, size, levels):
        super(LFSREncoder, self).__init__()
        self.flatten = torch.nn.Flatten()
        self.position = embeddings.Random(size * size, out_features)
        self.value = embeddings.Level(levels, out_features, high=256)
        #### my levels
        levels = []
        for number in range(NUM_LEVELS):
            my_list = []
            if number == 0:
                my_list = [-1]*(DIMENSIONS)
            elif number == NUM_LEVELS-1:
                my_list = [1]*(DIMENSIONS)
            else:
                mm = [-1]*(NUM_LEVELS-(number)) + [1]*(number)
                for i in range(c):
                    my_list = mm + my_list
                if  r!= 0:
                    my_list = mm[-r::] + my_list
            
            levels.append(my_list)
        arr = np.array(levels)
        tArr = torch.nn.Parameter(torchhd.MAPTensor(torch.from_numpy(arr).float()))
        self.value.weight = tArr.float()

        self.init_num = random.randint(1, 2**out_features)
        XORs_num = random.randint(2**(out_features-1), 2**out_features)
        
        init = [eval(i) for i in [*bin(self.init_num)[2:]]]
        init.extend([0] * (out_features - len(init)))
        self.XORs = [i for i, x in enumerate(reversed([*bin(XORs_num)[2:]])) if x == '1']
        lfsr = LFSR(init, self.XORs)
        sequence_length = size * size
        self.generated_sequence = lfsr.generate_sequence(sequence_length)
        arr = np.array(self.generated_sequence)
        tArr = torch.nn.Parameter(torchhd.MAPTensor(torch.from_numpy(arr).float()))
        self.position.weight = tArr.float()
    

    def forward(self, x):
        x = self.flatten(x)
        sample_hv = torchhd.bind(self.position.weight, self.value(x))
        sample_hv = torchhd.multiset(sample_hv)
        positive = torch.tensor(1.0, dtype=sample_hv.dtype, device=sample_hv.device)
        negative = torch.tensor(-1.0, dtype=sample_hv.dtype, device=sample_hv.device)
        return torch.where(sample_hv > 0, positive, negative)

class BaseLevelEncoder(torch.nn.Module):
    def __init__(self, out_features, size, levels):
        super(BaseLevelEncoder, self).__init__()
        self.flatten = torch.nn.Flatten()
        self.position = torchhd.embeddings.Random(size * size, out_features)
        self.value = torchhd.embeddings.Level(levels, out_features)
        self.name="BaseLevelEncoder"
        
    def forward(self, x):
        x = self.flatten(x)
        x = torchhd.bind(self.position.weight, self.value(x)).to(x.device)
        x = torchhd.multiset(x)
        return torchhd.hard_quantize(x)

if d["LFSR"]:
    encode = LFSREncoder(DIMENSIONS, IMG_SIZE, NUM_LEVELS)
else:
    encode = BaseLevelEncoder(DIMENSIONS, IMG_SIZE, NUM_LEVELS)
encode = encode.to(device)

num_classes = len(train_ds.classes)
model = Centroid(DIMENSIONS, num_classes)
model = model.to(device)
shadow_weight = None
trained_weight = None


def batch_encode_dataset(loader, batch_size=ENCODE_BATCH_SIZE):
    """Pre-encode an entire dataset in large batches for GPU throughput.

    Returns (all_hv, all_labels) tensors on *device*.
    """
    encode_ld = torch.utils.data.DataLoader(
        loader.dataset, batch_size=batch_size, shuffle=False
    )
    all_hv = []
    all_labels = []
    with torch.no_grad():
        for samples, labels in tqdm(encode_ld, desc="Encoding"):
            samples = samples.to(device)
            hv = encode(samples)
            all_hv.append(hv)
            all_labels.append(labels)
    return torch.cat(all_hv, dim=0), torch.cat(all_labels, dim=0).to(device)


def train():
    with torch.no_grad():
        for samples, labels in tqdm(train_ld, desc="Training"):
            samples = samples.to(device)
            labels = labels.to(device)

            samples_hv = encode(samples)
            model.add(samples_hv, labels)
    global shadow_weight
    if shadow_weight is None:
        shadow_weight = model.weight.detach().clone()
    torch.save(model.weight,                path+"/model/int_weights.pt")

    # Print first 10 values of each element in trained_weight
    #print("First 10 values of each trained weight element:")
    #for i, w in enumerate(trained_weight):
    #    print(f"Class {i}: {w[:10]}")

def test():
    accuracy = torchmetrics.Accuracy("multiclass", num_classes=num_classes)
    model.normalize(quantize=True)
    with torch.no_grad():
        count = 0
        for samples, labels in tqdm(test_ld, desc="Testing"):
            count = count + 1
            samples = samples.to(device)
            samples_hv = encode(samples)
            outputs = model(samples_hv, dot=True)
            accuracy.update(outputs.cpu(), labels)

    print(f"Testing accuracy of {(accuracy.compute().item() * 100):.3f}%")

    torch.save(model.weight,                path+"/model/chvs.pt")
    torch.save(encode.position.weight,      path+"/model/BV.pt")
    torch.save(encode.value.weight,         path+"/model/ID.pt")
    if isinstance(encode, LFSREncoder):
        torch.save(encode.init_num,             path+"/model/init_num.pt")
        torch.save(encode.XORs,                 path+"/model/xors.pt")
        torch.save(encode.generated_sequence,   path+"/model/sequence.pt")

def online_learning(epochs: int = 1, lr: int = 64, loader=None):
    """Quantization-aware online updates using the binary-weight error signal."""
    global shadow_weight
    if shadow_weight is None:
        # Try loading persisted shadow weights from a previous online-learning run,
        # falling back to the initial training weights (int_weights.pt).
        shadow_path = path + "/model/shadow_weight.pt"
        int_path = path + "/model/int_weights.pt"
        if os.path.isfile(shadow_path):
            print("Loading shadow weights from previous online-learning run")
            shadow_weight = torch.load(shadow_path, map_location=device)
        elif os.path.isfile(int_path):
            print("Loading initial training weights (int_weights.pt)")
            shadow_weight = torch.load(int_path, map_location=device)
        else:
            shadow_weight = model.weight.detach().clone()
    # restore shadow (full-precision) weights before online updates
    model.weight = torch.nn.Parameter(shadow_weight.clone().to(device), requires_grad=False)

    ld = loader if loader is not None else train_ld
    # Pre-encode entire dataset in large batches, then update one sample at a time
    all_hv, all_labels = batch_encode_dataset(ld)
    with torch.no_grad():
        for epoch in range(epochs):
            for i in tqdm(range(all_hv.size(0)), desc=f"Online (QA) epoch {epoch+1}"):
                model.add_online_quantized_aware(all_hv[i:i+1], all_labels[i:i+1], lr)

    # keep the updated shadow weights for future reuse
    shadow_weight = model.weight.detach().clone()
    # persist shadow weights and updated model weights to disk
    torch.save(shadow_weight, path + "/model/shadow_weight.pt")
    torch.save(model.weight,  path + "/model/int_weights.pt")

def online_learning_standard(epochs: int = 1, lr: float = 64.0, sim: str = "cos", loader=None):
    """Online updates using the standard OnlineHD error-corrective rule (add_online)."""

    ld = loader if loader is not None else train_ld
    # Pre-encode entire dataset in large batches, then update one sample at a time
    all_hv, all_labels = batch_encode_dataset(ld)
    with torch.no_grad():
        for epoch in range(epochs):
            for i in tqdm(range(all_hv.size(0)), desc=f"Online epoch {epoch+1}"):
                model.add_online(all_hv[i:i+1], all_labels[i:i+1], lr, False)

    # persist updated model weights to disk
    torch.save(model.weight, path + "/model/int_weights.pt")

