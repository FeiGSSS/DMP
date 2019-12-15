import pickle as pkl
import torch as T
import numpy as np
from torch_scatter import scatter_add

class IC():

    def __init__(self, net_path, seed_list):
        with open(net_path, "rb") as f:
            edge_list = pkl.load(f)

        self.src_nodes = T.LongTensor(edge_list[0])
        self.tar_nodes = T.LongTensor(edge_list[1])
        self.weights   = T.FloatTensor(edge_list[2])

        self.N = max([T.max(self.src_nodes), T.max(self.tar_nodes)]).item() + 1
        self.E = len(self.src_nodes)
        self.seeds = T.zeros(self.N); self.seeds[seed_list] = 1

        self.active = self.seeds
        self.new_active = self.active

        self.log = []

    def forward(self):
        random_P = T.rand(self.E)
        success = self.new_active[self.src_nodes] * (random_P < self.weights).float()
        success_active = (scatter_add(success, self.tar_nodes) >= 1).float()

        active_all = ((self.active + success_active) >= 1).float()

        self.new_active = active_all - self.active
        self.active = active_all

    def run(self):
        while self.new_active.sum() > 0:
            self.log.append(self.active.sum().item())
            self.forward()

        return self.log

def MC_IC_cpu(net_path, seed_list, mc):
    active_log = []

    for i in range(mc):
        ic = IC(net_path, seed_list)
        active_log.append(ic.run())

    max_steps = max([len(log) for log in active_log])
    active_log_max = [log+[log[-1]]*(max_steps-len(log)) for log in active_log]

    return np.mean(np.array(active_log_max), axis=0)

def MC_IC(net_path, seed_list, mc, nproc=40):
    active_log = []
    from multiprocessing import Pool
    pool = Pool(nproc)

    res = []
    for c in range(nproc):
        res.append(pool.apply_async(MC_IC_cpu, (net_path, seed_list, int(mc/nproc))))
    pool.close()
    pool.join()

    final_res = []
    for re in res:
        final_res.append(re.get())
    
    return np.mean(np.array(final_res), axis=0)


