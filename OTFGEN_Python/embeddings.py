import torch
import torchhd
import random

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

class LFSREncoder(torch.nn.Module):
    def __init__(self, out_features, channels, size, levels):
        super(LFSREncoder, self).__init__()
        self.name="LFSREncoder"
        self.flatten = torch.nn.Flatten(start_dim=-2)
        self.position = torchhd.embeddings.Random(size, out_features)
        self.value = torchhd.embeddings.Level(levels, out_features)
        levels_l = []
        for number in range(levels):
            if number == 0:
                my_list = [-1]*(out_features)
            elif number == levels-1:
                my_list = [1]*(out_features)
            else:
                my_list = [-1]*(out_features-(number*channels)) + [1]*(number*channels)
            my_list = my_list[::-1]
            levels_l.append(my_list)
        arr = torch.tensor(levels_l)
        print(arr.shape)
        tArr = torch.nn.Parameter(torchhd.MAPTensor(arr.float()))
        self.value.weight = tArr.float()
        XORs_num = random.randint(2**(out_features-1), 2**out_features)
        #init_num = random.randint(1, 2**out_features)
        #init = [eval(i) for i in [*bin(init_num)[2:]]]
        #init.extend([0] * (out_features - len(init)))
        init = torch.randint(0,2,(out_features,))
        XORs = [i for i, x in enumerate(reversed([*bin(XORs_num)[2:]])) if x == '1']
        lfsr = LFSR(init, XORs)
        sequence_length = size
        generated_sequence = lfsr.generate_sequence(sequence_length)
        arr = torch.tensor(generated_sequence)
        tArr = torch.nn.Parameter(torchhd.MAPTensor(arr.float()))
        self.position.weight = tArr.float()

    def forward(self, x):
        x = self.flatten(x)
        sample_hv = torchhd.bind(self.position.weight, self.value(x))
        sample_hv = torchhd.multiset(sample_hv)
        return torchhd.hard_quantize(sample_hv)
    

class RandomEncoder(torch.nn.Module):
    def __init__(self, out_features, channels, size, levels):
        super(RandomEncoder, self).__init__()
        self.flatten = torch.nn.Flatten(start_dim=-2)
        self.project = torchhd.embeddings.Sinusoid(size * size * channels, out_features)
        self.name="RandomEncoder"

    def forward(self, x):
        x = self.flatten(x)
        x = self.project(x)
        #x = torchhd.multiset(x)
        return torchhd.hard_quantize(x)

class PermutationEncoder(torch.nn.Module):
    def __init__(self, out_features, channels, size, levels):
        super(PermutationEncoder, self).__init__()
        self.flatten = torch.nn.Flatten(start_dim=-1)
        self.project = torchhd.embeddings.Level(levels, out_features)
        # self.project_r = torchhd.embeddings.Level(levels, out_features)
        # self.project_g = torchhd.embeddings.Level(levels, out_features)
        # self.project_b = torchhd.embeddings.Level(levels, out_features)

    def forward(self, x):
        x = self.flatten(x)
        # x_r = self.project_r(x[:,0])
        # x_g = self.project_g(x[:,1])
        # x_b = self.project_b(x[:,2])
        x_r = self.project(x[:,0])
        x_g = self.project(x[:,1])
        x_b = self.project(x[:,2])
        x = torchhd.bind_sequence(torch.stack([x_r, x_g, x_b], dim=1))
        # x = torchhd.multibind(torch.stack([x_r, x_g, x_b], dim=1))
        return torchhd.hard_quantize(x)

class BaseLevelEncoder(torch.nn.Module):
    def __init__(self, out_features, channels, size, levels):
        super(BaseLevelEncoder, self).__init__()
        self.flatten = torch.nn.Flatten(start_dim=-2)
        self.position = torchhd.embeddings.Random(size * size * channels, out_features)
        self.value = torchhd.embeddings.Level(levels, out_features)
        self.name="BaseLevelEncoder"
        
    def forward(self, x):
        x = self.flatten(x)
        x = torchhd.bind(self.position.weight, self.value(x)).to(x.device)
        x = torchhd.multiset(x)
        return torchhd.hard_quantize(x)
    
