from gem.embedding.hope import HOPE
from gem.embedding.lap import LaplacianEigenmaps
from WGE.lle import lle_cupy as lles
from WGE.DeepWalk import DeepWalk
from karateclub import MNMF
from ge import LINE
from node2vec import Node2Vec
import numpy as np
import networkx as nx

from clusim.clustering import Clustering

from WGE.utils import save_scores_to_csv
from WGE.utils import save_to_csv

def perform_hope_embedding(graph, embedding_dimension, _):
    hope_model = HOPE(d=embedding_dimension, beta=0.01)
    embd = hope_model.learn_embedding(graph=graph, is_weighted=False, no_python=True)
    return embd

def perform_laplacian_embedding(graph, embedding_dimension, _):
    lap_model = LaplacianEigenmaps(d=embedding_dimension)
    embd = lap_model.learn_embedding(graph=graph, is_weighted=False, no_python=True)
    return embd

def perform_lle_embedding(graph, embedding_dimension, _):
    embd = lles(graph, embedding_dimension)
    return embd

def perform_deepwalk_embedding(graph, embedding_dimension, _):
    model = DeepWalk(dimensions=embedding_dimension, walk_length=40, window_size=10, walk_number=80, workers=2)
    model.fit(graph)
    embd = model.get_embedding()
    return embd

def perform_mnmf_embedding(graph, embedding_dimension, number_of_intrinsic_clusters):
    graph_copy = graph.copy()
    H = nx.relabel.convert_node_labels_to_integers(graph_copy)
    MNMF_model = MNMF(dimensions=embedding_dimension, clusters=number_of_intrinsic_clusters, 
                      lambd=0.2, alpha=0.05, beta=0.05, iterations=200, lower_control=1e-15, eta=5.0, seed=42)
    MNMF_model.fit(H)
    embd = MNMF_model.get_embedding()
    return embd


def perform_line_embedding(graph, embedding_dimension, _):
    model = LINE(graph, embedding_size=embedding_dimension, order='first')
    model.train(batch_size=8192, epochs=100, verbose=0)
    LINE_embd = model.get_embeddings()
    embd = list(LINE_embd.values())
    return embd

def perform_node2vec_embedding(graph, embedding_dimension,_):
    # Precompute probabilities and generate walks - **ON WINDOWS ONLY WORKS WITH workers=1**
    print("mmmmmm")
    node2vec_model = Node2Vec(graph, dimensions=embedding_dimension, walk_length=10, num_walks=80, workers=32, quiet=True)
    # Embed nodes 
    print("flag4")
    #node2vec_fit = node2vec_model.fit(window=10, min_count=1, batch_words=80000)
    node2vec_fit = node2vec_model.fit(window=10, min_count=1, batch_words=20000)  
    print("flag5")
    # Any keywords acceptable by gensim.Word2Vec can be passed, `dimensions` and `workers` are automatically passed 
    # (from the Node2Vec constructor)
    embd = []
    print("flag6")
    for i in range(graph.number_of_nodes()):
        embd.append(node2vec_fit.wv[str(i)])
    return embd


def Embed_Graph(method: int, graph, embedding_dimension, number_of_intrinsic_clusters):
    labels = ["1HOPE", "2LAP", "3LLE", "4DeepWalk", "5MNMF", "6LINE", "7Node2Vec"]
    #print(labels[method-1])
    
    embedding_methods = {
        1: (perform_hope_embedding, "HOPE"),
        2: (perform_laplacian_embedding, "LAP"),
        3: (perform_lle_embedding, "LLE"),
        4: (perform_deepwalk_embedding, "DeepWalk"),
        5: (perform_mnmf_embedding, "MNMF"),
        6: (perform_line_embedding, "LINE"),
        7: (perform_node2vec_embedding, "Node2Vec")
    }

    embedding_func, method_label = embedding_methods[method]

    embd = embedding_func(graph, embedding_dimension, number_of_intrinsic_clusters)
    return embd



from WGE.eval_embd import eval_embd as EE
from networkx.generators.community import LFR_benchmark_graph
import concurrent.futures

dim_rg = [16,32,64,128,256];

avg_deg_seq = [25, 36, 42, 42, 42, 46, 46, 46, 48, 50]
min_comm_seq = [80, 160, 208, 208, 208, 210, 215, 210, 220, 240]
#avg_deg_seq = [36, 42, 42, 42, 46, 46, 46, 48, 50]
#min_comm_seq = [160, 208, 208, 208, 210, 215, 210, 220, 240]

def OneRound(method_id, itera = 5):
    RECORD = []
    for i in range(1,itera+1):
        SCORES = []
        for n in range(1000,10001,1000):
            print("flag1")
            tau1 = 2  # Power-law exponent for the degree distribution
            tau2 = 1.1 # Power-law exponent for the community size distribution should be >1
            mu = 0.01 # Mixing parameter
            avg_deg = avg_deg_seq[int(n/1000)-1] # Average Degree
            max_deg = int(n/10) # Max Degree
            min_commu = min_comm_seq[int(n/1000)-1] # Min Community Size
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

            # 存储 list 和 clustering 格式的拷贝 省得以后需要再做类型转换了
            intrinsic_list = list(intrinsic_membership.values())
            number_of_intrinsic_clusters = len(np.unique(intrinsic_membership))
            from clusim.clustering import Clustering
            intrinsic_clustering = Clustering(elm2clu_dict={i: [intrinsic_membership[i]] for i in intrinsic_membership.keys()})
            
            scores=[]
            print("flag2")
            for D in dim_rg:
                #print("flag")
                embd = Embed_Graph(method_id, G, D, max(intrinsic_list)+1)
                print("flag3")
                scores.append(EE(max(intrinsic_list)+1, intrinsic_list, intrinsic_clustering, embd))
            SCORES.append(scores)
        RECORD.append(SCORES)
    return RECORD
