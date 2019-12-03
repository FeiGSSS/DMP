import numpy as np
import pickle as pkl

data_name = "baseline"

src_nodes = np.array([0,0,0,0,1,1,1,1])
tar_nodes = np.array([2,3,4,5,8,9,6,7])
weights   = np.array([0.2]*len(src_nodes))

cave = []
edge_tuple = [(s,t) for s, t in zip(src_nodes, tar_nodes)]
E = len(edge_tuple)

for edge in edge_tuple:
    for j, st in enumerate(edge_tuple):
        if edge == st[::-1]:
            cave.append(j)
            break
        elif j == E-1:
            cave.append(E)
cave = np.array(cave)
data_np = np.array([src_nodes, tar_nodes, weights, cave])

with open("./"+data_name+".npy", "wb") as f:
    pkl.dump(data_np, f)
