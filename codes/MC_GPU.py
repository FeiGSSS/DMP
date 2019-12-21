import pickle as pkl
import numpy as np
import torch as T
import torch.nn as nn
import torch.multiprocessing as mp
from torch_scatter import scatter_add

class IC(nn.Module):

    def __init__(self, net_path,  device):
        super(IC, self).__init__()
        with open(net_path, "rb") as f:
            edge_list = pkl.load(f)

        self.src_nodes = T.LongTensor(edge_list[0]).to(device)
        self.tar_nodes = T.LongTensor(edge_list[1]).to(device)
        self.weights   = T.FloatTensor(edge_list[2]).to(device)

        self.N = max([T.max(self.src_nodes), T.max(self.tar_nodes)]).item() + 1
        self.E = len(self.src_nodes)

        self.device = device

    def _set_seed(self, seed_list):
        self.seeds = T.zeros(self.N, device=self.device)
        self.seeds[seed_list] = 1.0
        self.active = self.seeds
        self.new_active = self.active
        self.log = []
        

    def _spread(self):
        random_P = T.rand(self.E, device=self.device)
        success = self.new_active[self.src_nodes] * (random_P < self.weights).float()
        success_active = (scatter_add(success, self.tar_nodes) >= 1).float()

        active_all = ((self.active + success_active) >= 1).float()

        self.new_active = active_all - self.active
        self.active = active_all

    def run(self, seed_list):
        # Initial
        self._set_seed(seed_list)
        # Iteration
        while self.new_active.sum() > 0:
            self.log.append(self.active.sum().item())
            self._spread()

        return self.log

def MC_RUN(net_path, seed_list, mc, device):
    active_log = []
    ic_model = IC(net_path, device)
    ic_model = ic_model.to(device)
    
    for i in range(mc):
        active_log.append(ic_model.run(seed_list))

    max_steps = max([len(log) for log in active_log])
    active_log_max = [log+[log[-1]]*(max_steps-len(log)) for log in active_log]

    return np.mean(np.array(active_log_max), axis=0)[-1]
