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

# 这个函数将 list 或者 dict 类型的聚类结果进行类型转换
from clusim.clustering import Clustering

def to_clus(input):
    if isinstance(input, list):
        elm2clu_dict = {i: [input[i]] for i in range(len(input))}
    elif isinstance(input, dict):
        elm2clu_dict = {i: [input[i]] for i in input.keys()}
    else:
        raise ValueError("Input must be a list or a dictionary.")

    return Clustering(elm2clu_dict)


import clusim.sim as sim
def ECSim(clus1, clus2): #ECsim函数可以接受的参数类型可以是 dic, list, 或者 Clustering
    if not isinstance(clus1, Clustering):
        clus1 = to_clus(clus1)
    if not isinstance(clus2, Clustering):
        clus2 = to_clus(clus2)     
    return sim.element_sim(clus1, clus2, alpha=0.9)