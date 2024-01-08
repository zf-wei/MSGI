from auxpack.evaluate_embd import evaluate_embd as EE
import numpy as np
from karateclub import MNMF
import networkx as nx
from networkx.generators.community import LFR_benchmark_graph

dim_rg = [16,32,64,128,256];

RECORD = []

for i in range(1,51):
    SCORES = []
    for n in range(1000,10001,1000):
        tau1 = 2  # Power-law exponent for the degree distribution
        tau2 = 1.1 # Power-law exponent for the community size distribution should be >1
        mu = 0.5 # Mixing parameter
        avg_deg = 25 # Average Degree
        max_deg = int(n/10) # Max Degree
        min_commu = 60 #int(0.02*n) # Min Community Size
        max_commu = int(n/10) # Max Community Size

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

        scores=[]

        for D in dim_rg:

            # Create an instance of the MNMF model
            MNMF_model = MNMF(dimensions = D, clusters = max(intrinsic_list)+1, lambd = 0.2, 
                         alpha = 0.05, beta = 0.05, iterations = 100, 
                         lower_control = 1e-15, eta = 5.0, seed = 42)

            # Fit the model to the graph
            MNMF_model.fit(G)

            # Obtain the graph embeddings
            MNMF_embd = MNMF_model.get_embedding()

            scores.append(EE(intrinsic_list, intrinsic_clustering, MNMF_embd))
        SCORES.append(scores)
    RECORD.append(SCORES)


RECORD = np.array(RECORD)
for i in [0,1,2,3]:
    np.savetxt(f"MNMF0.5_{i}.txt", np.mean(RECORD, axis=0)[:,:,i])