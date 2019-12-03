from functools import reduce
import numpy as np
from mc_ic import MC_IC

import torch as T
import torch
import torch.nn.functional as F
import torch.optim as optim
from dmp_ic import DMP_IC

import time
t0 = time.time()

T.cuda.set_device(1)

net_path = "../data/test_graph/NetHEPT.npy"
IC = DMP_IC(net_path)

G = IC.G
over_rate = 0.8

def find_top_K(idx, K):

    top_k = []
    covered_tree = set()
    for node in idx.tolist():
        #print(len(covered_tree), len(top_k))
        node_nei1 = list(G.neighbors(node))
        node_nei2 = [list(G.neighbors(node)) for node in node_nei1]
        node_nei  = set(reduce(lambda x, y:x+y, node_nei2+[node_nei1]))
        # how many neighbors not been covered yet 
        if len(node_nei-covered_tree)/len(node_nei) < over_rate:
            continue
        else:
            covered_tree = covered_tree.union(node_nei)
            top_k.append(node)
            if len(top_k) >= K:
                break


    return top_k

def Penal_K(K, lr=0.01, iter=100, rand=False, eval_step=True):
    INF = []
    idx = T.argsort(IC.out_d, descending=True)
    seed_list = find_top_K(idx, K)
    inf = MC_IC(net_path, seed_list, mc=1000)[-1]
    Seed_tensor = T.zeros(IC.N); Seed_tensor[seed_list] = 1
    Sigmas = IC.run(Seed_tensor)
    print("Inf  = {:.2f}, DMP_Inf = {:.2f}".format(inf, Sigmas[-1].item()))
    INF.append(inf.item())

    return seed_list, INF

penal_log_10_lr = []
for k in range(1, 51, 2):
    lr_log = []
    for lr in [1E-3]:
        print("***", k, "***", lr)
        lr_log.append(Penal_K(k, lr=lr, iter=10))
    penal_log_10_lr.append(lr_log)
    
penal_log_10_lr_sigma = [[max(log_lr[1]) for log_lr in log_k] for log_k in penal_log_10_lr]
penal_log_10_lr_sigma_max = [max(s) for s in penal_log_10_lr_sigma]

out_weight_degree = IC.out_weight_d
idx = T.argsort(out_weight_degree, descending=True)

import pickle as pkl

save_path = "../results/lr_hept_overlip2_wd_{}".format(over_rate)

with open("{}.pkl".format(save_path), "wb") as f:
    pkl.dump(penal_log_10_lr_sigma, f)
with open("{}_max.pkl".format(save_path), "wb") as f:
    pkl.dump(penal_log_10_lr_sigma_max, f)

print(save_path, " saved!")
