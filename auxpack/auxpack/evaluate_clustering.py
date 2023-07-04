### 计算 NMI
# 我自己设计的函数可以接受 list 或者 dictl 类型的输入
# 内在的社群结构 intrinsic_membership 和
# Louvain 算法给出的 louvain_membership 

from sklearn.metrics import normalized_mutual_info_score
def NMI(clus1, clus2): # NMI函数可以接受的参数类型可以是 dic 或者 list
    if isinstance(clus1, dict):
        clus1 = list(clus1.values())
    if isinstance(clus2, dict):
        clus2 = list(clus2.values())     
    return normalized_mutual_info_score(clus1, clus2)



## 计算 ECSim 的函数

from clusim.clustering import Clustering
from clusim.sim import element_sim
import numpy as np


def to_clus(input):
    if isinstance(input, Clustering):
        return input
    elif isinstance(input, dict):
        return Clustering({i: [input[i]] for i in input.keys()})
    elif isinstance(input, list) or isinstance(input, np.ndarray):
        return Clustering({i: [input[i]] for i in range(len(input))})
    else:
        raise ValueError("Input must be a dictionary, a listan, or an np.array.")

def ECSim(clus1, clus2):
    clus1 = to_clus(clus1)
    clus2 = to_clus(clus2)
    return element_sim(clus1, clus2, alpha=0.9)
