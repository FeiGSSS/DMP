import networkx as nx
import torch as T
from torch_scatter import scatter_mul, scatter_add
from torch_geometric.utils import degree
import pickle as pkl
import time

class DMP_IC():
    def __init__(self, net_path, device, threshold): 
        with open(net_path, "rb") as f:
            self.edge_list = pkl.load(f)
        # edge_list with size [3, E], (src_node, tar_node, weight) 
        self.device = device
        self.src_nodes = T.LongTensor(self.edge_list[0]).to(device)
        self.tar_nodes = T.LongTensor(self.edge_list[1]).to(device)
        self.weights   = T.FloatTensor(self.edge_list[2]).to(device)
        self.cave_index = T.LongTensor(self.edge_list[3]).to(device)
        
        self.N = max([T.max(self.src_nodes), T.max(self.tar_nodes)]).item()+1
        self.E = len(self.src_nodes)
        self.d = degree(self.tar_nodes, num_nodes=self.N).to(device)
        self.out_d = degree(self.src_nodes, num_nodes=self.N).to(device)
        self.out_weight_d = scatter_add(self.weights, self.src_nodes).to(device)

        self.G = nx.DiGraph()
        self.G.add_edges_from(self.edge_list[:2].T)

        self.threshold = threshold

    def _set_seeds(self, seed_list):
        self.seeds = seed_list if T.is_tensor(seed_list) else T.Tensor(seed_list)
        self.seeds = self.seeds.to(self.device)
        self.Ps_i_0 = 1 - self.seeds

        self.Theta_0 = T.ones(self.E).to(self.device)        # init Theta(t=0)
        self.Ps_0 = 1 - self.seeds[self.src_nodes]    # Ps(t=0)
        self.Phi_0 = 1 - self.Ps_0 # init Thetau(t=0)

        self.Theta_t = self.Theta_0 - self.weights * self.Phi_0 + 1E-10 #get rid of NaN
        self.Ps_t_1 = self.Ps_0             # Ps(t-1)
        self.Ps_t = self.Ps_0 * self.mulmul(self.Theta_t) # Ps(t)
        self.inf_log = [self.seeds.sum(), self.influence()]

    def mulmul(self, Theta_t):
        Theta = scatter_mul(Theta_t, index=self.tar_nodes) # [N]
        Theta = Theta[self.src_nodes] #[E]
        Theta_cav = scatter_mul(Theta_t, index=self.cave_index)[:self.E]

        mul = Theta / Theta_cav
        return mul

    def forward(self):
        Phi_t = self.Ps_t_1 - self.Ps_t
        self.Theta_t = self.Theta_t - self.weights * Phi_t
        Ps_new = self.Ps_0 * self.mulmul(self.Theta_t)

        self.Ps_t_1 = self.Ps_t
        self.Ps_t   = Ps_new
    
    def influence(self):
        # Ps_i : the probability of node i being S 
        self.Ps_i = self.Ps_i_0 * scatter_mul(self.Theta_t, index=self.tar_nodes)
        return sum(1-self.Ps_i)
        
    def run(self, seed_list):
        self._set_seeds(seed_list)
        while True:
            self.forward()
            new_inf = self.influence()

            if abs(new_inf - self.inf_log[-1]) < self.threshold:
                break
            else:
                self.inf_log.append(new_inf)

        return self.inf_log

    def theta_aggr(self):
        theta = scatter_mul(self.Theta_t, index=self.tar_nodes)

        return theta, self.Ps_i
