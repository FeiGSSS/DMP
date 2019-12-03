import sys
sys.path.append("..")

from codes.mc_ic import MC_IC
from codes.dmp_ic import DMP_IC
import pickle as pkl
import torch


data = "twitter"
net_path = "../data/test_graph/{}.npy".format(data)
IC = DMP_IC(net_path=net_path)

with open("{}_imm_range.pkl".format(data), "rb") as f:
    seed_list = pkl.load(f)
    
    

dmp_mc_re = []
for seed in seed_list:
    tmp = []
    # DMP
    S = torch.zeros(IC.N); S[seed]=1
    tmp.append(IC.run(S)[-1].item())
    
    # MC
    tmp.append(MC_IC(net_path, seed, mc=1000)[-1])
    
    dmp_mc_re.append(tmp)
    
    print(len(seed))
    
save_path = "../results/{}_imm.pkl".format(data)
with open(save_path, "wb") as f:
    pkl.dump(dmp_mc_re, f)
    
print("saved to ", save_path)
print(dmp_mc_re)
