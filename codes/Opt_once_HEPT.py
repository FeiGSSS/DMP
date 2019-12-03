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
over_rate = 1

def find_top_K(idx, K):

    top_k = []
    covered_tree = set()
    for node in idx.tolist():
        #print(len(covered_tree), len(top_k))
        node_nei = set(list(G.neighbors(node)))
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
    S = T.zeros(IC.N, requires_grad=True)
    opt = optim.SGD([S], lr=lr)

    INF = []
    LOSS = []
    
    for i in range(1, iter):
        opt.zero_grad()
        Seed = T.sigmoid(S)
        Sigmas =IC.run(Seed)

        penal = T.pow(Seed.sum(), 2)
        loss = -Sigmas[-1] + penal
        #print("Loss = {:.2f}, Penal = {:.2f}".format(loss.item(), penal.item()))
        loss.backward()
        opt.step()
        
        LOSS.append(loss.item())

        if eval_step: 
            Seed = T.sigmoid(S)
            idx = T.argsort(Seed, descending=True)
            seed_list = find_top_K(idx, 1) ## CHECK!
            inf = MC_IC(net_path, seed_list, mc=1000)[-1]
            if len(INF)>=1 and inf<INF[-1]:
                break
            Seed_tensor = T.zeros(IC.N); Seed_tensor[seed_list] = 1
            Sigmas = IC.run(Seed_tensor)
            print("Inf  = {:.2f}, DMP_Inf = {:.2f}".format(inf, Sigmas[-1].item()))
            #print("DMP_Inf = {:.2f}".format(Sigmas[-1].item()))
            INF.append(inf.item())

    # Select All
    Seed = T.sigmoid(S)
    idx = T.argsort(Seed, descending=True)
    seed_list = find_top_K(idx, K)

    K_nodes_inf = []
    print("MC begin......")
    for i in range(1, K, 2):
        inf = MC_IC(net_path, seed_list[:i], mc=1000)[-1]
        K_nodes_inf.append(inf.item())

    print(K_nodes_inf)
    return K_nodes_inf

lr_log = []
for lr in [1E-3, 1E-4]:
    print("***", lr)
    lr_log.append(Penal_K(K=51, lr=lr, iter=10))
    
penal_log_10_lr_sigma = [[lr_log[0][i], lr_log[1][i]] for i in range(len(lr_log[0]))]
penal_log_10_lr_sigma_max = [max(s) for s in penal_log_10_lr_sigma]

out_weight_degree = IC.out_weight_d
idx = T.argsort(out_weight_degree, descending=True)

import pickle as pkl

save_path = "../results/lr_hept_opt_once_overlip_{}".format(over_rate)

with open("{}.pkl".format(save_path), "wb") as f:
    pkl.dump(penal_log_10_lr_sigma, f)
with open("{}_max.pkl".format(save_path), "wb") as f:
    pkl.dump(penal_log_10_lr_sigma_max, f)

print(save_path, " saved!")
