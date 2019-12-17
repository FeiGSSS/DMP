"""
从Overlap的结果中，选出DMP最大的种子集合，并用MC得到其近似传播范围。

"""

import os    
import time
import argparse
import numpy as np
import pickle as pkl
from mc_ic import MC_IC
    
def influence(data_name, results_tag):
    t0 = time.time()
    net_path = "../data/test_graph/{}.npy".format(data_name)
    res_path = "../results/{}/{}.pkl".format(data_name, results_tag)

    with open(res_path, "rb") as f:
        res = pkl.load(f)

    SeedListMax = []
    for re in res:
        seed_dmp = list(re.values())[0]
        dmp = [x[1] for x in seed_dmp]
        max_idx = np.argmax(dmp)
        SeedListMax.append(seed_dmp[max_idx][0])
        
    K_Seed = {len(seed): [seed] for seed in SeedListMax} 

    for seed in SeedListMax:
        IC = MC_IC(net_path, seed, mc=5000, nproc=40)
        K_Seed[len(seed)].append(IC[-1])
        print("Seed_Size={:<2}, Influence={:.2f}".format(len(seed), IC[-1]))

    save_path = "../results/{}/{}.influence.pkl".format(data_name, results_tag)
    with open(save_path, "wb") as f:
        pkl.dump(K_Seed, f)
    
    print("="*72)
    print("Total time = {:.2f}s".format(time.time()-t0))
    print("="*72)

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
    parser.add_argument("--Seed0", type=int, default=1)
    parser.add_argument("--Penal", type=int, default=1)
    args = parser.parse_args()

    Data_name = args.Data_name

    uniq_log = "_".join([str(k)+"_"+str(v) if v !=None else ""  for k, v in vars(args).items()]) 

    print(uniq_log, " evaluation begin~")
    influence(Data_name, uniq_log) 
