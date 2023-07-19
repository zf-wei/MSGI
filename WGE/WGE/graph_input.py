import networkx as nx
import json
import numpy as np
from WGE.remove_procedure import remove_procedure_index

def graph_input(N: int, MU: list, random_disturb:bool):

    graphs = {}
    membership = {}
    between = {}
    remove_procedure = {}
    index = {}

    for mu in MU:
        with open(f'graph_{N}_{mu}.edgelist', 'r') as file:
            lines = file.readlines()

        ### Process the lines and create a list of number pairs
        edge_list = []
        for line in lines:
            pair = tuple(map(int, line.strip().split()))
            edge_list.append(pair)

        ### 新建一个图 
        G = nx.Graph()
        ### 向图添加点和边
        sorted_nodes=sorted(set(range(N)))
        G.add_nodes_from(sorted_nodes)
        G.add_edges_from(edge_list)
        graphs[mu]=G

        ### Load Community Info
        membership_list = f'graph_{N}_{mu}.membership'
        membership[mu] = np.loadtxt(membership_list, dtype=int)
        
        ### Load Betweeness
        btwn_file = f'graph_{N}_{mu}.between'
        between[mu] = np.loadtxt(btwn_file)
        
        if random_disturb:
            with open(f'graph_{N}_{mu}.stoch_rmv', 'r') as file:
                remove_procedure[mu] = json.load(file)
        else:
            with open(f'graph_{N}_{mu}.btwn_rmv', 'r') as file:
                remove_procedure[mu] = json.load(file)

        index[mu] = remove_procedure_index(remove_procedure=remove_procedure[mu], num_nodes=N)

    return [graphs, membership, between, remove_procedure, index]