import numpy as np
from mc_ic import MC_IC

import torch as T
import torch.nn.functional as F
import torch.optim as optim
from dmp_ic import DMP_IC

import time
t0 = time.time()

net_path = "data/test_graph/twitter.npy"
IC = DMP_IC(net_path)

def Penal_K(K, lr=0.001, iter=8, rand=False):
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

        loss = -Sigmas[-1] + T.pow(Seed.sum()-K, 2)
        loss.backward()
        opt.step()
        
        LOSS.append(loss.item())
        
    idx = T.argsort(Seed, descending=True)
    seed_list = idx[:K]
        
    t_mc = time.time()
    print("MC begin...")
    #inf = MC_IC(net_path, seed_list.tolist(), mc=1000)[-1]
    Seed_tensor = T.zeros(IC.N); Seed_tensor(seed_list)
    inf = IC.run(Seed_tensor)[-1]
    print(inf, "MC time = {:.2f}".format(time.time()-t_mc))
    INF.append(inf)

    return seed_list, INF

print(Penal_K(10))

"""
penal_log_10_lr = []
for k in range(1, 51, 2):
    lr_log = []
    for lr in [1E-1, 1E-2, 1E-3, 1E-4, 1E-5, 1E-6]:
        print("***", k, "***", lr)
        lr_log.append(Penal_K(k, lr=lr, iter=10))
    penal_log_10_lr.append(lr_log)
    
penal_log_10_lr_sigma = [[max(log_lr[1]) for log_lr in log_k] for log_k in penal_log_10_lr]
penal_log_10_lr_sigma_max = [max(s) for s in penal_log_10_lr_sigma]

out_weight_degree = IC.out_weight_d
idx = T.argsort(out_weight_degree, descending=True)

wd_log = []
for k in range(1, 51, 2):
    seed_list = idx[:k]
    wd_log.append([seed_list, MC_IC(net_path, seed_list.tolist(), mc=1000)[-1]])

import pickle as pkl

with open("lr_sigma.pkl", "wb") as f:
    pkl.dump(penal_log_10_lr_sigma, f)
with open("lr_sigma_max.pkl", "wb") as f:
    pkl.dump(penal_log_10_lr_sigma_max, f)
with open("wd_log.pkl", "wb") as f:
    pkl.dump(wd_log, f)
"""
