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
from MC_GPU import MC_RUN
from mc_ic import MC_IC
from utils import find_top_K, Greedy_K

def Optimize(args, net_path, eval_step=True) :
    S = T.zeros(IC.N, requires_grad=True, device=device_opt)
    opt = optim.SGD([S], lr=args.Lr)

    
    for i in range(1, args.Iter):
        tts = time.time()

        opt.zero_grad()
        Seed = T.sigmoid(S)
        Sigmas = IC.run(Seed)
        loss = -Sigmas[-1] + T.pow(Seed.sum(), 2)
        loss.backward()
        opt.step()

        tte = time.time()
        print("Iter={} Loss={:.2f}, Opt_Time={:.2f}s".format(i, loss.item(), tte-tts)) 

    SeedSize_Inf = {}
    idx = T.argsort(S, descending=True)
    Seed_Prepar = find_top_K(G, idx, 51, overlap=args.OverLap, uncover_rate=args.OverLapRate)
    print("Eval Start ...... ")
    for K in range(Seed_range[0], Seed_range[1], Seed_range[2]):
        ts = time.time()
        Mc_Inf = MC_RUN(net_path, Seed_Prepar[:K], mc=1000, device=device_eva)
        te = time.time()
        print("{} : opt={:.1f} eval_time={:.1f}s".format(K, Mc_Inf, te-ts))
        SeedSize_Inf[K] = [Seed_Prepar[:K], Mc_Inf]
    print("")

    return SeedSize_Inf


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--Data_name", type=str, help="graph's name")
    parser.add_argument("--Lr", type=float, help="learning rate")
    parser.add_argument("--Iter", type=int, help="maximum iteration times of optim")
    parser.add_argument("--Max_Iter", type=int, help="maximum iteration times of DMP")
    parser.add_argument("--OverLap", type=int,  help="OverLap order list")
    parser.add_argument("--OverLapRate", type=float,  help="OverLap rate list")
    parser.add_argument("--Seed_range", type=int,  nargs="+")
    parser.add_argument("--Device_opt", type=int,  default=-1, help="-1 for cpu")
    parser.add_argument("--Device_eva", type=int,  default=2, help="-1 for cpu")
    args = parser.parse_args()

    Data_name = args.Data_name
    Lr = args.Lr
    Iter = args.Iter
    Max_Iter = args.Max_Iter
    OverLap = args.OverLap
    OverLapRate = args.OverLapRate
    Seed_range = args.Seed_range
    range_str = "_".join([str(x) for x in Seed_range])
    
    uniq_log = str(Data_name)+"_"+"_".join([str(par) for par in [Lr, Iter, Max_Iter, OverLap, OverLapRate]])+"_"+range_str
    
    net_path = "../data/test_graph/{}.npy".format(Data_name)
    if not os.path.exists("../results/{}".format(Data_name)):
        os.mkdir("../results/{}".format(Data_name))
    
    save_path = "../results/{}/{}.pkl".format(Data_name, uniq_log)

    #device_opt = T.device("cpu") if args.Device_opt==-1 else T.device("cuda:{}".format(args.Device_opt))
    device_opt = T.device("cpu")
    device_eva = T.device("cuda:{}".format(args.Device_eva))

    print("Preparing DMP module for {} .. ".format(Data_name))
    ts = time.time()
    IC = DMP_IC(net_path, T.device("cpu"), Max_Iter)
    G = IC.G
    print("Done, cost {:.1f}s".format(time.time()-ts))

    print("*"*72)
    print(uniq_log)
    print("*"*72)
    SeedSize_Inf = Optimize(args, net_path)
    with open(save_path, "wb") as f:
        pkl.dump(SeedSize_Inf, f)
