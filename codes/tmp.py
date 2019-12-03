from dmp_ic import DMP_IC
import torch as T
import time

net_path = "../data/test_graph/twitter.npy"
device = T.device("cpu")
IC = DMP_IC(net_path, device=device)

t0 = time.time()

seed = T.zeros(IC.N); seed[5666] = 1
seed = seed.to(device)

IC.run(seed)

print(time.time()-t0)
