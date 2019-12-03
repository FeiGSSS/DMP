from functools import reduce
import numpy as np
from collections import defaultdict

import torch as T
import torch
import torch.nn.functional as F
import torch.optim as optim
from dmp_ic import DMP_IC

import time
t0 = time.time()

T.cuda.set_device(1)

net_path = "../data/test_graph/NetHEPT.npy"
IC = DMP_IC(net_path, device=T.device("cpu"))

G = IC.G

def find_top_K(idx, K, overlap, uncover_rate):

    top_k = []
    if overlap==0:
        top_k = idx[:K].tolist()
    elif overlap==1:
        covered_tree = set()
        for node in idx.tolist():
            node_nei1 = list(G.neighbors(node))
            node_nei  = set(node_nei1)
            if len(node_nei-covered_tree)/len(node_nei) >= uncover_rate:
                covered_tree = covered_tree.union(node_nei)
                top_k.append(node)
                if len(top_k) >= K:
                    break
            else:
                continue
    elif overlap==2:
        covered_tree = set()
        for node in idx.tolist():
            node_nei1 = list(G.neighbors(node))
            node_nei2 = [list(G.neighbors(node)) for node in node_nei1]
            node_nei  = set(reduce(lambda x, y:x+y, node_nei2+[node_nei1]))
            # how many neighbors not been covered yet 
            if len(node_nei-covered_tree)/len(node_nei) >= uncover_rate:
                covered_tree = covered_tree.union(node_nei)
                top_k.append(node)
                if len(top_k) >= K:
                    break
            else:
                continue

    else:
        print(overlap, "Not implement......")
        exit()

    return top_k

def Penal_K(K, lr=0.0001, iter=20, rand=False, eval_step=True):
    S = T.zeros(IC.N, requires_grad=True)
    opt = optim.SGD([S], lr=lr)

    BEST_INF = 0
    Seed_Results = defaultdict(list)
    
    for i in range(1, iter):
        opt.zero_grad()
        Seed = T.sigmoid(S)
        Sigmas =IC.run(Seed)

        penal = T.pow(Seed.sum()-K, 2)
        loss = -Sigmas[-1] + penal
        loss.backward()
        opt.step()
        
        if eval_step: 
            Step_Inf =[]

            Seed = T.sigmoid(S)
            idx = T.argsort(Seed, descending=True)
            # K=0
            Seed0 = find_top_K(idx, K, overlap=0, uncover_rate=0)
            _Seed0 = T.zeros(IC.N); _Seed0[Seed0] = 1
            INF0 = IC.run(_Seed0)[-1].item()
            Step_Inf.append(INF0)
            Seed_Results["0"].append([Seed0, INF0])

            # K>0
            for overlap in [1, 2]:
                for uncover_rate in [0.3, 0.5, 0.8, 1]:
                    Seed = find_top_K(idx, K, overlap=overlap, uncover_rate=uncover_rate)
                    _Seed = T.zeros(IC.N); _Seed[Seed] = 1
                    Inf = IC.run(_Seed)[-1].item()
                    Step_Inf.append(Inf)

                    Seed_Results[str(overlap)+"_"+str(uncover_rate)].append([Seed, Inf])

            Step_Inf_Max = max(Step_Inf)
            print("DMP_Step_Inf_Max = {:.2f}".format(Step_Inf_Max))

            if Step_Inf_Max <= BEST_INF:
                #break
                pass
            else:
                BEST_INF = Step_Inf_Max

    return Seed_Results

Results_Dict = []
for k in range(1, 51, 2):
    print(">>>>>>{}<<<<<<".format(k))
    Results_Dict.append(Penal_K(k))
     
import pickle as pkl

save_path = "../results/NetHEPT_overlap_Lr_1E-4_dmp_10"

with open("{}.pkl".format(save_path), "wb") as f:
    pkl.dump(Results_Dict, f)

print(save_path, " saved!")
