import pickle as pkl
from codes.mc_ic import MC_IC
with open("results/twitter_overlip0_SeedList.pkl", "rb") as f:
    twitter_overlap0_SeedList = pkl.load(f)
twitter_seed = [s[0] for s in twitter_overlap0_SeedList]
#twitter_overlip2_0_5_SeedList_lr_1E_3 = [s[0] for s in twitter_overlip2_0_5_SeedList]
#twitter_overlip2_0_5_SeedList_lr_1E_4 = [s[1] for s in twitter_overlip2_0_5_SeedList]

net_path = "data/test_graph/twitter.npy"

re = []

for s in twitter_seed:
    re.append(MC_IC(net_path, s, mc=1000)[-1])
    print(len(s), re[-1])

with open("results/twitter_overlip0_MC1000.pkl", "wb") as f:
    pkl.dump(re, f)
