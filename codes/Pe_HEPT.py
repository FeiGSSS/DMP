import numpy as np
from mc_ic import MC_IC

import torch as T
import torch.nn.functional as F
import torch.optim as optim
from dmp_ic import DMP_IC

import time
t0 = time.time()

T.cuda.set_device(1)

net_path = "../data/test_graph/NetHEPT.npy"
IC = DMP_IC(net_path)

def Penal_K(K, lr=0.01, iter=100, rand=False, eval_step=True):
    S = T.zeros(IC.N, requires_grad=True)
    opt = optim.SGD([S], lr=lr)

    INF = []
    LOSS = []
    LOSS_min = 1E+10
    Patient = 10
    
    for i in range(1, iter):
        opt.zero_grad()
        Seed = T.sigmoid(S)
        Sigmas =IC.run(Seed)

        penal = T.pow(Seed.sum()-K, 2)
        loss = -Sigmas[-1] + penal
        loss.backward()
        
        # manual grad
        #print(S.grad)
        #print(S)
        #S.grad += 1E+5*(Seed * (1-Seed)).detach()
        #print(S.grad)
        #print(S)

        opt.step()
        
        LOSS.append(loss.item())
        if eval_step: 
            Seed = T.sigmoid(S)
            idx = T.argsort(Seed, descending=True)
            seed_list = idx[:K]
            inf = MC_IC(net_path, seed_list.tolist(), mc=1000)[-1]

            Seed_tensor = T.zeros(IC.N); Seed_tensor[seed_list] = 1
            Sigmas = IC.run(Seed_tensor)
            print("Inf  = {:.2f}, DMP_Inf = {:.2f}".format(inf, Sigmas[-1].item()))
            INF.append(inf)

    return seed_list, INF

Penal_K(15, lr=1E-4, iter=10)
