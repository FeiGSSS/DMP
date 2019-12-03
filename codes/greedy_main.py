import torch as T
import torch.nn.functional as F
import torch.optim as optim
from dmp_ic import DMP_IC

import time
t0 = time.time()

T.cuda.set_device(1)

#net_path = "data/NetHEPT.npy"
net_path = "data/BA_100.npy"
#net_path = "data/baseline.npy"
IC = DMP_IC(net_path)

K=10
N = IC.N
Seeds = []
old_inf = 0

for i in range(K):
    aug_inf = -1
    new_node = -1
    for n in range(N):
        if n not in Seeds:
            S = T.zeros(N)
            S[Seeds+[n]] = 1
            inf = IC.run(S)[-1].item()
            if inf-old_inf > aug_inf:
                new_node = n
                aug_inf = inf-old_inf
        #if (n+1)%10 == 0:
        #    print(n, "time={:.2f}s".format(time.time()-t0))
    Seeds += [new_node]
    old_inf += aug_inf

from mc_ic import MC_IC
import numpy as np

print("Seeds=", Seeds)
print("Inf={:.2f}".format(MC_IC(net_path, Seeds, mc=1000)[-1]))
    
