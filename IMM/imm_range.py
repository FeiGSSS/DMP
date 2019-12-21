import os
import pickle as pkl

data = "LiveJournal1"
path = "../data/graph_txt/{}.txt".format(data)
seed_list = []
for k in range(1, 51, 2):
    re = os.popen("python IMP.py -i {} -k {}".format(path, k)).readlines()
    seed = re[0][2:][:-2].split(",")
    seed = [int(i) for i in seed]
    seed_list.append(seed)

    print(k, seed_list[-1])

import pickle as pkl

with open("{}_imm_range.pkl".format(date), "wb") as f:
    pkl.dump(seed_list, f)
