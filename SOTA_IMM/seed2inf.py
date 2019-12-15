import sys
sys.path.append("..")

from codes.mc_ic import MC_IC
import pickle as pkl

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--Data_name", type=str, help="graph's name")
args = parser.parse_args()

data_name = args.Data_name
net_path = "../data/test_graph/{}.npy".format(data_name)

with open("{}_imm_range.pkl".format(data_name), "rb") as f:
    seed_list = pkl.load(f)

res = {}
for seed in seed_list:
    infs = MC_IC(net_path, seed, mc=5000, nproc=40)
    res[len(seed)] = [seed, infs[-1]]
    print(len(seed), infs[-1])
    
save_path = "./{}_imm_infs.pkl".format(data_name)
with open(save_path, "wb") as f:
    pkl.dump(res, f)
print("saved to ", save_path)
