import torch.nn as nn
import torch.nn.functional as F
import torch
import sklearn
import sklearn.cluster
from torch_geometric.nn import GCNConv

class GCN(torch.nn.Module):
    def __init__(self, nfeat, nhid, nout):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(nfeat, nhid)
        self.conv2 = GCNConv(nhid, nout)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)

        return F.log_softmax(x, dim=1)

def cluster(data, k, temp, num_iter, init = None, cluster_temp=5):
    '''
    pytorch (differentiable) implementation of soft k-means clustering.
    '''
    #normalize x so it lies on the unit sphere
    data = torch.diag(1./torch.norm(data, p=2, dim=1)) @ data
    #use kmeans++ initialization if nothing is provided
    if init is None:
        data_np = data.detach().numpy()
        norm = (data_np**2).sum(axis=1)
        init = sklearn.cluster.k_means_._k_init(data_np, k, norm, sklearn.utils.check_random_state(None))
        init = torch.tensor(init, requires_grad=True)
        if num_iter == 0: return init
    mu = init
    n = data.shape[0]
    d = data.shape[1]
#    data = torch.diag(1./torch.norm(data, dim=1, p=2))@data
    for t in range(num_iter):
        #get distances between all data points and cluster centers
#        dist = torch.cosine_similarity(data[:, None].expand(n, k, d).reshape((-1, d)), mu[None].expand(n, k, d).reshape((-1, d))).reshape((n, k))
        dist = data @ mu.t()
        #cluster responsibilities via softmax
        r = torch.softmax(cluster_temp*dist, 1)
        #total responsibility of each cluster
        cluster_r = r.sum(dim=0)
        #mean of points in each cluster weighted by responsibility
        cluster_mean = (r.t().unsqueeze(1) @ data.expand(k, *data.shape)).squeeze(1)
        #update cluster means
        new_mu = torch.diag(1/cluster_r) @ cluster_mean
        mu = new_mu
    dist = data @ mu.t()
    r = torch.softmax(cluster_temp*dist, 1)
    return mu, r, dist


class GCNClusterNet(nn.Module):
    '''
    The ClusterNet architecture. The first step is a 2-layer GCN to generate embeddings.
    The output is the cluster means mu and soft assignments r, along with the 
    embeddings and the the node similarities (just output for debugging purposes).
    
    The forward pass inputs are x, a feature matrix for the nodes, and adj, a sparse
    adjacency matrix. The optional parameter num_iter determines how many steps to 
    run the k-means updates for.
    '''
    def __init__(self, nfeat, nhid, nout, K, cluster_temp=10):
        super(GCNClusterNet, self).__init__()

        self.GCN = GCN(nfeat, nhid, nout)
        self.distmult = nn.Parameter(torch.rand(nout))
        self.sigmoid = nn.Sigmoid()
        self.K = K
        self.cluster_temp = cluster_temp
        self.init =  torch.rand(self.K, nout)
        
    def forward(self, data, num_iter=10):
        embeds = self.GCN(data)
        mu_init, _, _ = cluster(embeds, self.K, 1, num_iter, cluster_temp = self.cluster_temp, init = self.init)
        mu, r, dist = cluster(embeds, self.K, 1, 1, cluster_temp = self.cluster_temp, init = mu_init.detach().clone())
        x = torch.softmax(dist * 10, 0).sum(dim=1)
        x = 2*(torch.sigmoid(4*x) - 0.5)
        if x.sum() > self.K:
            x = self.K*x/x.sum()
        return x

class GCNOnly(nn.Module):
    def __init__(self, nfeat, nhid, nout):
        super(GCNOnly, self).__init__()

        self.gcn = GCN(nfeat, nhid, nout)
        self.decode = nn.Linear(nout, 1)
        
    def forward(self, data):
        embeds = self.gcn(data)
        pred = torch.sigmoid(self.decode(embeds))

        return pred
