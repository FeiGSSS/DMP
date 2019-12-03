import networkx as nx
import numpy as np
import pickle as pkl

data_name = "BA_500"

n = 500
m = 3
w = 0.2


G = nx.barabasi_albert_graph(n, m)
edges = list(G.edges)

src_nodes = np.array([e[0] for e in edges])
tar_nodes = np.array([e[1] for e in edges])
weights = np.array([w]*len(src_nodes))

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
