import sys
sys.path.append("..")

from codes.MC_GPU import MC_RUN
import pickle as pkl
import torch as T

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--Data_name", type=str, help="graph's name")
parser.add_argument("--Device_id", type=int, help="id of cuda")
parser.add_argument("--MC", type=int, help="num of simulation")
parser.add_argument("--range", type=int, nargs="+")
args = parser.parse_args()
range_str = "_".join([str(x) for x in args.range])
data_name = args.Data_name

net_path = "../data/test_graph/{}.npy".format(data_name)

with open("{}_imm_range_{}.pkl".format(data_name, range_str), "rb") as f:
    seed_list = [item[0] for item in list(pkl.load(f).values())]

res = {}
for seed in seed_list:
    infs = MC_RUN(net_path, seed, mc=args.MC, device=T.device("cuda:{}".format(args.Device_id)))
    res[len(seed)] = [seed, infs]
    print(len(seed), infs)
    
save_path = "./{}_imm_infs_{}.pkl".format(data_name, range_str)
with open(save_path, "wb") as f:
    pkl.dump(res, f)
print("saved to ", save_path)
