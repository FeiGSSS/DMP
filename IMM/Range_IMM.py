import os
import pickle as pkl
import argparse
import time

parser = argparse.ArgumentParser()
parser.add_argument("--Data_name", type=str, default="NetHEPT")
parser.add_argument("--Seed_range", type=int, nargs="+")
args = parser.parse_args()
range_str = "_".join([str(x) for x in args.Seed_range])

path = "../data/graph_txt/{}.txt".format(args.Data_name)
seed_list = {}
for k in range(args.Seed_range[0], args.Seed_range[1], args.Seed_range[2]):
    ts = time.time()
    re = os.popen("python IMP.py -i {} -k {}".format(path, k)).readlines()
    te = time.time()
    seed = re[0][1:][:-2].split(",")
    assert k == len(seed)
    seed = [int(i.strip()) for i in seed]
    seed_list[k] = [seed, te-ts]

    print(k, te-ts)

with open("../SOTA_IMM/{}_imm_range_{}.pkl".format(args.Data_name, range_str), "wb") as f:
    pkl.dump(seed_list, f)
