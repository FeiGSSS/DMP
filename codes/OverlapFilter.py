"""
从Overlap的结果中，选出DMP最大的种子集合，并用MC得到其近似传播范围。

"""

import pickle as pkl

results_tag = "../results/Twitter_overlap_Lr_1E-4_dmp_10"
#net_path = "../data/test_graph/NetHEPT.npy"
net_path = "../data/test_graph/twitter.npy"

with open(results_tag+".pkl", "rb") as f:
    re = pkl.load(f)

keys = ["0"]
for i in [1, 2]:
    for j in [0.3, 0.5, 0.8, 1]:
        keys.append(str(i)+"_"+str(j))

K_Seed = {}
for log in re:
    K = len(log["0"][0][0])
    Seed = []
    MAX_INF = 0
    for key in keys:
        seed_inf = log[key]
        for s, i in seed_inf:
            if i>MAX_INF:
                Seed = s
                MAX_INF = i
    K_Seed[K] = Seed

from mc_ic import MC_IC

Seed_INF = {}
for k , v in K_Seed.items():
    IC = MC_IC(net_path, seed_list = v, mc=1000)
    Seed_INF[k] = IC[-1]
    print(k, IC[-1])

save_path = results_tag+"_max.pkl"
with open(save_path, "wb") as f:
    pkl.dump(Seed_INF, f)
