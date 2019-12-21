import torch as T

from collections import defaultdict
from functools import reduce

def find_top_K(G, idx, K, overlap, uncover_rate):

    top_k = []
    if overlap==0:
        top_k = idx[:K].tolist()
    elif overlap==1:
        covered_tree = set()
        for i, node in enumerate(idx.tolist()):
            node_nei1 = list(G.neighbors(node))
            node_nei  = set(node_nei1)
            if len(node_nei-covered_tree)/len(node_nei) >= uncover_rate:
                covered_tree = covered_tree.union(node_nei)
                top_k.append(node)
                if len(top_k) >= K:
                    break
            else:
                continue
    elif overlap==2:
        covered_tree = set()
        for i, node in enumerate(idx.tolist()):
            node_nei1 = list(G.neighbors(node))
            node_nei2 = [list(G.neighbors(node)) for node in node_nei1]
            node_nei  = set(reduce(lambda x, y:x+y, node_nei2+[node_nei1]))
            # how many neighbors not been covered yet 
            if len(node_nei-covered_tree)/len(node_nei) >= uncover_rate:
                covered_tree = covered_tree.union(node_nei)
                top_k.append(node)
                if len(top_k) >= K:
                    break
            else:
                continue

    else:
        print(overlap, "Not implement......")
        exit()

    return top_k 

def Greedy_K(DMP, device, sorted_nodes, K):
    "从排名靠前的2K个节点中，用DMP+Greedy选K个"
    Sorted_Nodes = sorted_nodes.tolist()[:2*K]
    Nodes_Margin = []
    for node in Sorted_Nodes:
        _seed_tensor = T.zeros(DMP.N, device=device)
        _seed_tensor[node] = 1
        Nodes_Margin.append([node, DMP.run(_seed_tensor)[-1].item()])
    Nodes_Margin = sorted(Nodes_Margin, key=lambda x:x[1], reverse=True) 
    Seed_K = [Nodes_Margin[0][0]]
    Seed_K_Inf = Nodes_Margin[0][1]
    Nodes_Margin.pop(0)

    while len(Seed_K) <= K:
        Updated_Nodes = []
        while True:
            current_node = Nodes_Margin[0][0]
            if current_node in Updated_Nodes:
                Seed_K.append(current_node)
                break
            else:
                Updated_Nodes.append(current_node)

            tmp_seed = Seed_K + [current_node]
            tmp_seed_tensor = T.zeros(DMP.N, device=device)
            tmp_seed_tensor[tmp_seed] = 1
            current_node_margin = DMP.run(tmp_seed_tensor)[-1].item() - Seed_K_Inf  
            Nodes_Margin[0][1] = current_node_margin
            if Nodes_Margin[0][1] >= max([item[1] for item in Nodes_Margin]):
                Nodes_Margin.pop(0)
                Seed_K.append(current_node)
                break
            else:
                Nodes_Margin = sorted(Nodes_Margin, key=lambda x:x[1], reverse=True) 
                

    return Seed_K
    



