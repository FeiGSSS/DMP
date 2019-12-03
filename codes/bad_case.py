import numpy as np
from mc_ic import MC_IC

import torch as T
import torch.nn.functional as F
import torch.optim as optim
from dmp_ic import DMP_IC

import time
t0 = time.time()

T.cuda.set_device(1)

net_path = "../data/test_graph/bad_case.npy"
IC = DMP_IC(net_path)

def Penal_K(K, lr=0.01, iter=100, rand=False, eval_step=True):
    S = T.zeros(IC.N, requires_grad=True)
    opt = optim.SGD([S], lr=lr)

    INF = []
    
    print(S)
    for i in range(1, iter):
        opt.zero_grad()
        Seed = T.sigmoid(S)
        Sigmas =IC.run(Seed)
        print(Sigmas)
        penal = T.pow(Seed.sum()-K, 2)
        loss = -Sigmas[-1] #+ penal
        loss.backward()
        print("*"*72)
        print("Grad:", S.grad)
        opt.step()
        print("S Value:",  S)
        print("*"*72)
        
        if eval_step: 
            idx = T.argsort(Seed, descending=True)
            seed_list = idx[:K]
            inf = MC_IC(net_path, seed_list.tolist(), mc=1000)[-1]

            Seed_tensor = T.zeros(IC.N); Seed_tensor[seed_list] = 1
            Sigmas = IC.run(Seed_tensor)
            print("Inf  = {:.2f}, DMP_Inf = {:.2f}".format(inf, Sigmas[-1].item()))
            INF.append(inf)

    # Final...
    Seed = T.sigmoid(S)
    idx = T.argsort(Seed, descending=True)
    seed_list = idx[:K].tolist()
    INF.append(MC_IC(net_path, seed_list, mc=1000)[-1])
    return seed_list, INF

seed, inf = Penal_K(2, lr=1E-4, iter=10)
print(seed, inf)
