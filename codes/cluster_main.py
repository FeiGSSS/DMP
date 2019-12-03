import torch as T
import torch.optim as optim
from dmp_ic import DMP_IC
from model import GCNClusterNet
from torch_geometric.data import Data

lr = 0.05
K = 10
model_cluster = GCNClusterNet(nfeat=1,
            nhid=30,
            nout=30,
            K = K,
            cluster_temp = 1)

#T.cuda.set_device(1)
net_path = "data/NetHEPT.npy"
#net_path = "data/baseline.npy"
IC = DMP_IC(net_path)
data = Data(x=IC.d.unsqueeze(1), edge_index=T.stack((IC.src_nodes, IC.tar_nodes)))

#data = data.to("cuda:1")
#model_cluster = model_cluster.to("cuda:1")

opt = optim.Adam(model_cluster.parameters(), lr=lr)

print("lr={}, K={}, data={}".format(lr, K, net_path))

for i in range(100):
    opt.zero_grad()
    Seed = model_cluster(data)
    Sigmas = IC.run(Seed)
    theta, Ps_i = IC.theta_aggr()
    degree = IC.d
    Grid_seed = -(1-degree)*theta
    Seed.backward(gradient=Grid_seed)
    #loss = - Sigmas[-1]
    #loss.backward()
    
    print(max(Seed), min(Seed))
    opt.step()
    print(Sigmas[-1].item(), len(Sigmas))

Seed = model_cluster(data)
argsort = T.argsort(Seed, descending = True)

final_seed = T.zeros(IC.N); final_seed[argsort[:K]] = 1
final_sigma= IC.run(final_seed)
print("Final Seeds : ", argsort[:K])
print(final_sigma)

