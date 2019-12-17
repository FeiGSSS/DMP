import os
import time
import argparse
import numpy as np
import pickle as pkl
from functools import reduce
from collections import defaultdict

import torch as T
import torch.nn.functional as F
import torch.optim as optim

from dmp_ic import DMP_IC

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

def Penal_K(K, lr, iter, OverLap, OverLapRate, Penal=True, eval_step=True):
    S = T.zeros(IC.N, requires_grad=True)
    opt = optim.SGD([S], lr=lr)

    S = S.to(device)
    BEST_INF = 0
    Seed_Results = defaultdict(list)
    
    for i in range(1, iter):
        tts = time.time()
        opt.zero_grad()
        Seed = T.sigmoid(S)
        Sigmas =IC.run(Seed)

        penal = T.pow(Seed.sum()-K, 2)
        if Penal:	
            loss = -Sigmas[-1] + penal
        else:
            loss = -Sigmas[-1]
        loss.backward()
        opt.step()
        tte = time.time()
        
        tes = time.time()
        if eval_step: 
            Step_Inf =[]
            Seed = T.sigmoid(S)
            idx = T.argsort(Seed, descending=True)

            # K>0
            for overlap in OverLap:
                for uncover_rate in OverLapRate:
                    Seed = find_top_K(idx, K, overlap=overlap, uncover_rate=uncover_rate)
                    _Seed = T.zeros(IC.N); _Seed[Seed] = 1
                    Inf = IC.run(_Seed)[-1].item()
                    Step_Inf.append(Inf)

                    Seed_Results[str(overlap)+"_"+str(uncover_rate)].append([Seed, Inf])
            tee = time.time()
            Step_Inf_Max = max(Step_Inf)

            print("Loss={:.2f}, Inf_Max={:.2f}, Train_Time={:.2f}s, Eval_Time={:.2f}s".format(loss.item(), Step_Inf_Max, tte-tts, tee-tes))

            if Step_Inf_Max <= BEST_INF:
                #break
                pass
            else:
                BEST_INF = Step_Inf_Max

    return Seed_Results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--Data_name", type=str, help="graph's name")
    parser.add_argument("--Lr", type=float, help="learning rate")
    parser.add_argument("--Thr", type=float, help="threshold of stopping in DMP")
    parser.add_argument("--Space", type=int, help="step length of seed set size")
    parser.add_argument("--Iter", type=int, help="maximum iteration times of optim")
    parser.add_argument("--Max_Iter", type=int, help="maximum iteration times of DMP")
    parser.add_argument("--OverLap", type=int, nargs="+", help="OverLap order list")
    parser.add_argument("--OverLapRate", type=float,  nargs='+', help="OverLap rate list")
    parser.add_argument("--Seed0", type=int,  default=1, help="the smallest seed size")
    parser.add_argument("--Penal", type=int,  default=1, help="whether to use penality")
    args = parser.parse_args()

    Data_name = args.Data_name
    Lr = args.Lr
    Threshold = args.Thr
    Space = args.Space
    Iter = args.Iter
    Max_Iter = args.Max_Iter
    OverLap = args.OverLap
    OverLapRate = args.OverLapRate
    Seed0 = args.Seed0
    Penal = args.Penal
    
    uniq_log = "_".join([str(k)+"_"+str(v) for k, v in vars(args).items()])
    
    net_path = "../data/test_graph/{}.npy".format(Data_name)
    if not os.path.exists("../results/{}".format(Data_name)):
        os.mkdir("../results/{}".format(Data_name))
    
    save_path = "../results/{}/{}.pkl".format(Data_name, uniq_log)

    device = T.device("cpu")

    IC = DMP_IC(net_path, device, Threshold, Max_Iter)
    G = IC.G

    Results_Dict = []
    print("*"*72)
    print(uniq_log)
    print("*"*72)
    for k in range(Seed0, 51, Space):
        print(">>>>>>{}<<<<<<".format(k))
        Results_Dict.append(Penal_K(k, Lr, Iter, OverLap, OverLapRate, Penal))
        print("*"*20)
     

    with open(save_path, "wb") as f:
        pkl.dump(Results_Dict, f)

    print(save_path, " saved!")
