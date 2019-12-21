import numpy as np
import pickle as pkl
import time

t0 = time.time()

data_name = "LiveJournal1"

with open("./graph_txt/"+data_name+".txt", "r") as f:
    tri_lines = []
    for line in f.readlines():
        line = line.strip().split(" ")
        tri_lines.append(line)

src_nodes = []
tar_nodes = []
weights   = []

for s, t, w in tri_lines[1:]:
    src_nodes.append(int(s))
    tar_nodes.append(int(t))
    weights.append(float(w))

t1 = time.time()
print("Reading Done ... Time = {:.2f}s".format(t1-t0))

import networkx as nx
G = nx.DiGraph()

edge_list = [(s, t) for s, t in zip(src_nodes, tar_nodes)]
E = len(edge_list)

G.add_edges_from(edge_list)
attr = {edge:w for edge, w in zip(edge_list, range(E))}
nx.set_edge_attributes(G, attr, "idx")

cave = []

for edge in edge_list:
    if G.has_edge(*edge[::-1]):
        cave.append(G.edges[edge[::-1]]["idx"])
    else:
        cave.append(E)
print("Total time = {:.2f}s".format(time.time()-t1))
data_np = np.array([src_nodes, tar_nodes, weights, cave])

with open("./test_graph/"+data_name+".npy", "wb") as f:
    pkl.dump(data_np, f)
