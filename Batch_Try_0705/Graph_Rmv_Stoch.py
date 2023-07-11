### 使用 networkx 包中的函数 LFR_benchmark_graph 生成随机图
import networkx as nx
from networkx.generators.community import LFR_benchmark_graph

n = 1000
tau1 = 2  # Power-law exponent for the degree distribution
tau2 = 1.1 # Power-law exponent for the community size distribution 
            #S hould be >1
mu = 0.1 # Mixing parameter
avg_deg = 25 # Average Degree
max_deg = int(0.1*n) # Max Degree
min_commu = 60 # Min Community Size
max_commu = int(0.1*n) # Max Community Size

G0 = LFR_benchmark_graph(
    n, tau1, tau2, mu, average_degree=avg_deg, max_degree=max_deg, min_community=min_commu, max_community=max_commu, 
    seed=7
)
### 去掉 G 中的重边和自环 
G0 = nx.Graph(G0) # Remove multi-edges

selfloop_edges = list(nx.selfloop_edges(G0)) # a list of self loops

G0.remove_edges_from(selfloop_edges) # Remove self-loops

#################################################

import numpy as np
intrinsic_communities = {frozenset(G0.nodes[v]["community"]) for v in G0}
intrinsic_membership = np.empty(G0.number_of_nodes(), dtype=int)
for node in range(G0.number_of_nodes()):
    for index, inner_set in enumerate(intrinsic_communities):
        if node in inner_set:
            intrinsic_membership[node] = index
            break

#################################################

import os
from datetime import date
def generate_output(disturb: bool, filename):
    # Generate the folder name with the current date
    if disturb:   
        folder_name = f"Graph_Rmv_Stoch_{date.today()}"
    else:
        folder_name = f"Graph_Rmv_Btwn_{date.today()}"

    # Create the output directory if it doesn't exist
    output_dir = os.path.join(os.getcwd(), folder_name)
    os.makedirs(output_dir, exist_ok=True)

    # Construct the full file path
    file_path = os.path.join(output_dir, filename)
    
    return file_path

#################################################

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import StrMethodFormatter

def quar_plot(scores, disturb: bool, filename, win:10):
    data = scores
    # Extract x and y coordinates for each curve
    x = range(len(data))  # Use the length of data as x-coordinates
    y = list(zip(*data))  # Transpose the data matrix

    # Compute rolling mean with a window size of win
    y_smoothed = [pd.Series(curve).rolling(window=win, min_periods=1).mean() for curve in y]
    #print(y_smoothed)
    # Create subplots
    fig, axs = plt.subplots(len(y_smoothed), 1, sharex=True)

    # Plot each curve with smoothed data on a separate subplot
    ylabel = ["NMI Eucl", "NMI Cosn", "ECS Eucl", "ECS Cosn"]

    for i, curve in enumerate(y_smoothed):
        axs[i].plot(x, curve)
        axs[i].set_ylabel(ylabel[i]) 
        axs[i].yaxis.set_major_formatter(StrMethodFormatter('{x:.3f}'))
    
    # Add x-axis label to the last subplot
    axs[-1].set_xlabel('Number of Removed Vertices')

    # Adjust spacing between subplots
    plt.tight_layout()
    
    file_path = generate_output(disturb, filename+".png")
    plt.savefig(file_path)    
    
    # Show the plot
    plt.show()

#################################################

import os
import csv
from datetime import date

def save_scores_to_csv(scores, disturb: bool, filename):
    """
    Saves a list of 4-lists to a CSV file with a double space separator.
    
    Args:
        scores (list): The list of 4-lists to be saved.
        filename (str): The name of the output CSV file.
    """
    # Construct the full file path
    file_path = generate_output(disturb, filename+".csv")

    with open(file_path, 'w', newline='') as file:
        writer = csv.writer(file, delimiter=' ')
        for score_list in scores:
            writer.writerow(score_list)

#################################################

import random

temp = list(range(G0.number_of_nodes()))
random.shuffle(temp)

#################################################

from auxpack.eval_embd import eval_embd as EE
from clusim.clustering import Clustering

D=20
K = len(np.unique(intrinsic_membership))
wk=48
#Measure = []
#remain_nodes = np.array(range(G.number_of_nodes()))

#################################################

### 1 Hope 方法
from gem.embedding.hope import HOPE  

idx = [True] *(G0.number_of_nodes())
G=G0.copy()

scores=[]

for i in temp:
    G.remove_node(i)
    print(f"Vertex {i} is removed.", end=' ')
    if not nx.is_connected(G):
        print("In total", temp.index(i), "vertices are removed. G is now disconnected!!")
        break
    
    hope_model = HOPE(d=D, beta=0.01) 
    # A higher value of beta places more emphasis on capturing higher-order proximities
    embd = hope_model.learn_embedding(graph=G, is_weighted=False, no_python=True)
    
    idx[i] = False
    intrin_list = intrinsic_membership[idx]
    intrin_Clus = Clustering({i: [intrin_list[i]] for i in range(len(intrin_list))})
    K = len(np.unique(intrinsic_membership[idx]))
    score = EE(K,intrin_list,intrin_Clus, embd)
    scores.append(score)
    print("NMI&ECS:", score)
    
quar_plot(scores=scores, disturb=True, filename="1HOPE", win=10)
save_scores_to_csv(scores, True, "1HOPE")

#################################################

### 2 Laplacian 方法
from gem.embedding.lap import LaplacianEigenmaps

idx = [True] *(G0.number_of_nodes())
G=G0.copy()

scores=[]

for i in temp:
    G.remove_node(i)
    print(f"Vertex {i} is removed.", end=' ')
    if not nx.is_connected(G):
        print("In total", temp.index(i), "vertices are removed. G is now disconnected!!")
        break

    lap_model = LaplacianEigenmaps(d=D)
    embd = lap_model.learn_embedding(graph=G, is_weighted=False, no_python=True)
    
    idx[i] = False
    intrin_list = intrinsic_membership[idx]
    intrin_Clus = Clustering({i: [intrin_list[i]] for i in range(len(intrin_list))})
    K = len(np.unique(intrinsic_membership[idx]))
    score = EE(K,intrin_list,intrin_Clus, embd)
    scores.append(score)
    print("NMI&ECS:", score)
    
quar_plot(scores=scores, disturb=True, filename="2LAP", win=10)
save_scores_to_csv(scores, True, "2LAP")

#################################################

### 3 LLE 方法
from auxpack.lle import lle

idx = [True] *(G0.number_of_nodes())
G=G0.copy()

scores=[]

for i in temp:
    G.remove_node(i)
    print(f"Vertex {i} is removed.", end=' ')
    if not nx.is_connected(G):
        print("In total", temp.index(i), "vertices are removed. G is now disconnected!!")
        break
    
    embd = lle(G, D)
    
    idx[i] = False
    intrin_list = intrinsic_membership[idx]
    intrin_Clus = Clustering({i: [intrin_list[i]] for i in range(len(intrin_list))})
    K = len(np.unique(intrinsic_membership[idx]))
    score = EE(K,intrin_list,intrin_Clus, embd)
    scores.append(score)
    print("NMI&ECS:", score)
    
quar_plot(scores=scores, disturb=True, filename="3LLE", win=10)
save_scores_to_csv(scores, True, "3LLE")

#################################################

### 4 DeepWalk方法
from auxpack.DeepWalk import DeepWalk

idx = [True] *(G0.number_of_nodes())
G=G0.copy()

scores=[]

for i in temp:
    G.remove_node(i)
    print(f"Vertex {i} is removed.", end=' ')
    if not nx.is_connected(G):
        print("In total", temp.index(i), "vertices are removed. G is now disconnected!!")
        break
 
    model = DeepWalk(dimensions=D, walk_length=16, window_size=10, walk_number=10, workers=wk)
    model.fit(G)
    embd = model.get_embedding()
    
    idx[i] = False
    intrin_list = intrinsic_membership[idx]
    intrin_Clus = Clustering({i: [intrin_list[i]] for i in range(len(intrin_list))})
    K = len(np.unique(intrinsic_membership[idx]))
    score = EE(K,intrin_list,intrin_Clus, embd)
    scores.append(score)
    print("NMI&ECS:", score)
    
quar_plot(scores=scores, disturb=True, filename="4DeepWalk", win=10)
save_scores_to_csv(scores, True, "4DeepWalk")

#################################################

### 5 MNMF 方法
from karateclub import MNMF

idx = [True] *(G0.number_of_nodes())
G=G0.copy()

scores=[]

for i in temp:
    G.remove_node(i)
    print(f"Vertex {i} is removed.", end=' ')
    if not nx.is_connected(G):
        print("In total", temp.index(i), "vertices are removed. G is now disconnected!!")
        break

    # Create an instance of the MNMF model
    MNMF_model = MNMF(dimensions = D, clusters = K, lambd = 0.2, 
                 alpha = 0.05, beta = 0.05, iterations = 100, 
                 lower_control = 1e-15, eta = 5.0, seed = 42)

    # Fit the model to the graph
    H = nx.relabel.convert_node_labels_to_integers(G)
    MNMF_model.fit(H)
    # Obtain the graph embeddings
    embd = MNMF_model.get_embedding()
    
    idx[i] = False
    intrin_list = intrinsic_membership[idx]
    intrin_Clus = Clustering({i: [intrin_list[i]] for i in range(len(intrin_list))})
    K = len(np.unique(intrinsic_membership[idx]))
    score = EE(K,intrin_list,intrin_Clus, embd)
    scores.append(score)
    print("NMI&ECS:", score)
    
quar_plot(scores=scores, disturb=True, filename="5MNMF", win=10)
save_scores_to_csv(scores, True, "5MNMF")

#################################################

### 6 LINE 方法
from ge import LINE

idx = [True] *(G0.number_of_nodes())
G=G0.copy()

scores=[]

for i in temp:
    G.remove_node(i)
    print(f"Vertex {i} is removed.", end=' ')
    if not nx.is_connected(G):
        print("In total", temp.index(i), "vertices are removed. G is now disconnected!!")
        break
        
    model = LINE(G,embedding_size=D,order='first');
    model.train(batch_size=8192,epochs=50,verbose=0);# train model
    LINE_embd = model.get_embeddings();# get embedding vectors
    embd = list(LINE_embd.values())
    
    idx[i] = False
    intrin_list = intrinsic_membership[idx]
    intrin_Clus = Clustering({i: [intrin_list[i]] for i in range(len(intrin_list))})
    K = len(np.unique(intrinsic_membership[idx]))
    score = EE(K,intrin_list,intrin_Clus, embd)
    scores.append(score)
    print("NMI&ECS:", score)
    
quar_plot(scores=scores, disturb=True, filename="6LINE", win=10)
save_scores_to_csv(scores, True, "6LINE")

#################################################

### 7 Node2Vec 方法 以后使用这个
from node2vec import Node2Vec

nodes_range = np.array(range(G0.number_of_nodes()))
idx = [True] *(G0.number_of_nodes())
G=G0.copy()

scores=[]

for i in temp:
    G.remove_node(i)
    print(f"Vertex {i} is removed.", end=' ')
    if not nx.is_connected(G):
        print("In total", temp.index(i), "vertices are removed. G is now disconnected!!")
        break
        
    # Precompute probabilities and generate walks - **ON WINDOWS ONLY WORKS WITH workers=1**
    node2vec_model = Node2Vec(G, dimensions=D, walk_length=16, num_walks=10, workers=wk, quiet=True) #, temp_folder='test' # Use temp_folder for big graphs
    # Embed nodes 
    node2vec_fit = node2vec_model.fit(window=10, min_count=1, batch_words=20000)  
    # Any keywords acceptable by gensim.Word2Vec can be passed, `dimensions` and `workers` are automatically passed 
    # (from the Node2Vec constructor)

    idx[i] = False

    nodes = [str(x) for x in nodes_range[idx]]
    embd = np.array([node2vec_fit.wv[node] for node in nodes])
    
    intrin_list = intrinsic_membership[idx]
    intrin_Clus = Clustering({i: [intrin_list[i]] for i in range(len(intrin_list))})
    K = len(np.unique(intrinsic_membership[idx]))
    score = EE(K,intrin_list,intrin_Clus, embd)
    scores.append(score)
    print("NMI&ECS:", score)
    
quar_plot(scores=scores, disturb=True, filename="7Node2Vec", win=10)
save_scores_to_csv(scores, True, "7Node2Vec")

#################################################
