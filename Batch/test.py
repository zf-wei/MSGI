### 使用 networkx 包中的函数 LFR_benchmark_graph 生成随机图
import networkx as nx
from networkx.generators.community import LFR_benchmark_graph

n = 1000
tau1 = 2  # Power-law exponent for the degree distribution
tau2 = 1.1 # Power-law exponent for the community size distribution 
            #S hould be >1
mu = 0.05 # Mixing parameter
avg_deg = 25 # Average Degree
max_deg = 100 # Max Degree
min_commu = 50 # Min Community Size
max_commu = 100 # Max Community Size


G = LFR_benchmark_graph(
    n, tau1, tau2, mu, average_degree=avg_deg, max_degree=max_deg, min_community=min_commu, max_community=max_commu, 
    seed=2
)



### 去掉 G 中的重边和自环 
G = nx.Graph(G) # Remove multi-edges

selfloop_edges = list(nx.selfloop_edges(G)) # a list of self loops

G.remove_edges_from(selfloop_edges) # Remove self-loops


### LFR 图是有内在的社群结构的，每个节点的社群存储在其 community 属性中，是一个 set
# 通过运行循环，按照内在的社群结构给每个节点一个标签 即为其 intrinsic_membership
# 为了方便 intrinsic_membership 一开始是作为一个 dict 存储的
intrinsic_communities = {frozenset(G.nodes[v]["community"]) for v in G}
intrinsic_membership = {}
for node in range(G.number_of_nodes()):
    for index, inner_set in enumerate(intrinsic_communities):
        if node in inner_set:
            intrinsic_membership[node] = index
            break
# intrinsic_membership = list(intrinsic_membership.values())

# 存储 list 和 clustering 格式的拷贝 省得以后需要再做类型转换了
intrinsic_list = list(intrinsic_membership.values())
from clusim.clustering import Clustering
intrinsic_clustering = Clustering(elm2clu_dict={i: [intrinsic_membership[i]] for i in intrinsic_membership.keys()})


### 利用 Louvain 算法进行社群识别并画图
# louvain_membership 是作为一个 dict 给出的
from community import community_louvain

louvain_membership = community_louvain.best_partition(G)


### 利用 InfoMap 算法进行社群识别
# 输出类型为一个 list

# Convert the NetworkX graph to an igraph graph
import igraph as ig
iG = ig.Graph.from_networkx(G)

# Perform Infomap clustering using igraph, and get the membership as a list
infomap_membership = iG.community_infomap().membership # 类型为 list


### 导入计算 NMI 和 ECSim 的包 我自己封装的
from auxpack.evaluate_clustering import NMI
from auxpack.evaluate_clustering import ECSim as ECS

### 使用范例
print(NMI(louvain_membership, intrinsic_membership))
print(ECS(infomap_membership, intrinsic_membership))


### 导入 图嵌入评估函数 我自己封装的
from auxpack.evaluate_embedding import evaluate_embedding as EE



### 1 HOPE 方法
from gem.embedding.hope import HOPE

hope_model = HOPE(d=30, beta=0.01) 
# A higher value of beta places more emphasis on capturing higher-order proximities

hope_embd = hope_model.learn_embedding(graph=G, is_weighted=False, no_python=True)
print(EE(intrinsic_list, intrinsic_clustering, hope_embd))


### 2 Laplacian 方法
from gem.embedding.lap import LaplacianEigenmaps
lap_model = LaplacianEigenmaps(d=20)

lap_embd = lap_model.learn_embedding(graph=G, is_weighted=True, no_python=True)
print(EE(intrinsic_list, intrinsic_clustering, lap_embd))

### 3 MNMF 方法

from karateclub import MNMF

# Create an instance of the MNMF model
MNMF_model = MNMF(dimensions = 64, clusters = 14, lambd = 0.2, 
             alpha = 0.05, beta = 0.05, iterations = 100, 
             lower_control = 1e-15, eta = 5.0, seed = 42)

# Fit the model to the graph
MNMF_model.fit(G)

# Obtain the graph embeddings
MNMF_embd = MNMF_model.get_embedding()
print(EE(intrinsic_list, intrinsic_clustering, MNMF_embd))



### 4 Node2Vec 方法 

from node2vec import Node2Vec

# Precompute probabilities and generate walks - **ON WINDOWS ONLY WORKS WITH workers=1**
node2vec_model = Node2Vec(G, dimensions=64, walk_length=30, num_walks=50, workers=32) #, temp_folder='test' # Use temp_folder for big graphs
 
# Embed nodes 
node2vec_fit = node2vec_model.fit(window=10, min_count=1, batch_words=4096)  
# Any keywords acceptable by gensim.Word2Vec can be passed, `dimensions` and `workers` are automatically passed 
# (from the Node2Vec constructor)
# print("Embedding already generated!!")
node2vec_embd = []
for i in range(G.number_of_nodes()):
    node2vec_embd.append(node2vec_fit.wv[str(i)])
print(EE(intrinsic_list, intrinsic_clustering, node2vec_embd))



### 5 DeepWalk方法
from karateclub import DeepWalk
model = DeepWalk(dimensions=64, walk_length=30, window_size=10)
model.fit(G)
deepwalk_embd = model.get_embedding()
print(EE(intrinsic_list, intrinsic_clustering, deepwalk_embd))


### 6 LINE 方法
from ge import LINE
model = LINE(G,embedding_size=60,order='first');
model.train(batch_size=1024,epochs=50,verbose=0);# train model
LINE_embd = model.get_embeddings();# get embedding vectors

LINE_embd = list(LINE_embd.values())
print(EE(intrinsic_list, intrinsic_clustering, LINE_embd))


### 7 LLE 方法
from auxpack.lle import lle
D = 15
lle_embd = lle(G, D)
print(EE(intrinsic_list, intrinsic_clustering, lle_embd))