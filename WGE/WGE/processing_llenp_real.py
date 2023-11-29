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

from WGE.eval_embd import eval_embd as EE

import random
from itertools import combinations, groupby

def make_graph_connect(G):
    components = dict(enumerate(nx.connected_components(G)))
    components_combs = combinations(components.keys(), r=2)
    for _, node_edges in groupby(components_combs, key=lambda x: x[0]):
        node_edges = list(node_edges)
        random_comps = random.choice(node_edges)
        source = random.choice(list(components[random_comps[0]]))
        target = random.choice(list(components[random_comps[1]]))
        G.add_edge(source, target)


def perform_hope_embedding(graph, nodes_to_remove, embedding_dimension, _, __, ___):
    graph_copy = graph.copy()
    graph_copy.remove_nodes_from(nodes_to_remove)
    hope_model = HOPE(d=embedding_dimension, beta=0.01)
    if not nx.is_connected(graph_copy):
        make_graph_connect(graph_copy)
    embd = hope_model.learn_embedding(graph=graph_copy, is_weighted=False, no_python=True)
    return embd

def perform_laplacian_embedding(graph, nodes_to_remove, embedding_dimension, _, __, ___):
    graph_copy = graph.copy()
    graph_copy.remove_nodes_from(nodes_to_remove)
    lap_model = LaplacianEigenmaps(d=embedding_dimension)
    if not nx.is_connected(graph_copy):
        make_graph_connect(graph_copy)
    embd = lap_model.learn_embedding(graph=graph_copy, is_weighted=False, no_python=True)
    return embd

def perform_lle_embedding(graph, nodes_to_remove, embedding_dimension, _, __, ___):
    graph_copy = graph.copy()
    graph_copy.remove_nodes_from(nodes_to_remove)
    if not nx.is_connected(graph_copy):
        make_graph_connect(graph_copy)
    embd = lles(graph_copy, embedding_dimension)
    return embd

def perform_deepwalk_embedding(graph, nodes_to_remove, embedding_dimension, _, __, wk=32):
    graph_copy = graph.copy()
    graph_copy.remove_nodes_from(nodes_to_remove)
    model = DeepWalk(dimensions=embedding_dimension, walk_length=40, window_size=10, walk_number=80, workers=wk)
    if not nx.is_connected(graph_copy):
        make_graph_connect(graph_copy)
    model.fit(graph_copy)
    embd = model.get_embedding()
    return embd

def perform_mnmf_embedding(graph, nodes_to_remove, embedding_dimension, number_of_intrinsic_clusters, _, __):
    graph_copy = graph.copy()
    graph_copy.remove_nodes_from(nodes_to_remove)
    H = nx.relabel.convert_node_labels_to_integers(graph_copy)
    if not nx.is_connected(H):
        make_graph_connect(H)
    MNMF_model = MNMF(dimensions=embedding_dimension, clusters=number_of_intrinsic_clusters, 
                      lambd=0.2, alpha=0.05, beta=0.05, iterations=200, lower_control=1e-15, eta=5.0, seed=42)
    MNMF_model.fit(H)
    embd = MNMF_model.get_embedding()
    return embd


def perform_line_embedding(graph, nodes_to_remove, embedding_dimension, _, __, ___):
    graph_copy = graph.copy()
    graph_copy.remove_nodes_from(nodes_to_remove)
    if not nx.is_connected(graph_copy):
        make_graph_connect(graph_copy)
    model = LINE(graph_copy, embedding_size=embedding_dimension, order='first')
    model.train(batch_size=8192, epochs=100, verbose=0)
    LINE_embd = model.get_embeddings()
    embd = list(LINE_embd.values())
    return embd

def perform_node2vec_embedding(graph, nodes_to_remove, embedding_dimension,_, idx, wk=32):
    graph_copy = graph.copy()
    graph_copy.remove_nodes_from(nodes_to_remove)
    if not nx.is_connected(graph_copy):
        make_graph_connect(graph_copy)
    node2vec_model = Node2Vec(graph_copy, dimensions=embedding_dimension, walk_length=10, num_walks=80, workers=wk, quiet=True)
    node2vec_fit = node2vec_model.fit(window=10, min_count=1, batch_words=80000)
    nodes_range = np.array(range(graph.number_of_nodes()))
    nodes = [str(x) for x in nodes_range[idx]]
    embd = np.array([node2vec_fit.wv[node] for node in nodes])
    return embd


def calculate_score(embd, intrinsic_membership, number_of_intrinsic_clusters):
    intrin_list = intrinsic_membership
    intrin_Clus = Clustering({i: [intrin_list[i]] for i in range(len(intrin_list))})
    score = EE(number_of_intrinsic_clusters, intrin_list, intrin_Clus, embd)
    return score

def Comprehensive_Processing(output:bool, disturb_type:int, method: int, num_cpus:int, 
                             graph, embedding_dimension, intrinsic_membership, remove_procedure, remove_procedure_index_form, mu):
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

    number_of_intrinsic_clusters = len(np.unique(intrinsic_membership))
    idxx = np.ones(graph.number_of_nodes(), dtype=bool)
    embd = embedding_func(graph, [], embedding_dimension, number_of_intrinsic_clusters, idxx, num_cpus)
    score_0 = calculate_score(embd, intrinsic_membership[idxx], number_of_intrinsic_clusters)
    MEAN = [np.array(score_0)]
    
    STD = [np.array([0,0,0,0])]
    
    for rp, idx in zip(remove_procedure, remove_procedure_index_form):
        scores = []
        for nodes_to_remove, idxx in zip(rp, idx):
            number_of_intrinsic_clusters = len(np.unique(intrinsic_membership[idxx]))
            embd = embedding_func(graph, nodes_to_remove, embedding_dimension, number_of_intrinsic_clusters, idxx, num_cpus)
            score = calculate_score(embd, intrinsic_membership[idxx], number_of_intrinsic_clusters)
            scores.append(score)

        array = np.array(scores)
        mean = np.mean(array, axis=0)
        std = np.std(array, axis=0)
        MEAN.append(mean)
        STD.append(std)
        if output:
            save_scores_to_csv(disturb_type, scores, f"{graph.number_of_nodes()}_{mu}_{embedding_dimension}dim_" + labels[method-1] + "_SCORES")
            save_to_csv(disturb_type, MEAN, f"{graph.number_of_nodes()}_{mu}_{embedding_dimension}dim_" + labels[method-1] + "_MEAN")
            save_to_csv(disturb_type, STD, f"{graph.number_of_nodes()}_{mu}_{embedding_dimension}dim_" + labels[method-1] + "_STD")
    return MEAN, STD