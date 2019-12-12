from dmp_ic import DMP_IC
import torch
import time

net_path = "../data/test_graph/twitter.npy"
threshold = 100
cpu = torch.device("cpu")
cuda1 = torch.device("cuda:1")

IC_cpu = DMP_IC(net_path, cpu, threshold)
seed_list = torch.zeros(IC_cpu.N); seed_list[range(100)] = 1
t_cpu = time.time()
re = IC_cpu.run(seed_list)
print("Time1 = {:.2f}s".format(time.time()-t_cpu))

IC_gpu = DMP_IC(net_path, cuda1, threshold)
seed_list = torch.zeros(IC_gpu.N); seed_list[range(100)] = 1
t_gpu = time.time()
re = IC_gpu.run(seed_list)
print("Time2 = {:.2f}s".format(time.time()-t_gpu))
