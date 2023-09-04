import networkx as nx
import json
import numpy as np
from WGE.remove_procedure import remove_procedure_index

def graph_input(N: int, MU: list, disturb_type:int):

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
        
        if disturb_type==1:
            with open(f'graph_{N}_{mu}.stoch_rmv', 'r') as file:
                remove_procedure[mu] = json.load(file)
        elif disturb_type==2:
            with open(f'graph_{N}_{mu}.btwn_rmv', 'r') as file:
                remove_procedure[mu] = json.load(file)
        elif disturb_type==3:
            with open(f'graph_{N}_{mu}.trans_rmv', 'r') as file:
                remove_procedure[mu] = json.load(file)
        elif disturb_type==4:
            with open(f'graph_{N}_{mu}.deg_rmv', 'r') as file:
                remove_procedure[mu] = json.load(file)
        elif disturb_type==5:
            with open(f'graph_{N}_{mu}.rank_rmv', 'r') as file:
                remove_procedure[mu] = json.load(file)
        elif disturb_type==6:
            with open(f'graph_{N}_{mu}.trank_rmv', 'r') as file:
                remove_procedure[mu] = json.load(file)

        index[mu] = remove_procedure_index(remove_procedure=remove_procedure[mu], num_nodes=N)

    return [graphs, membership, between, remove_procedure, index]


    ############### 简易版本
def graph_input_simple(N: int, mu: list, disturb_type: int):

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
    sorted_nodes = sorted(set(range(N)))
    G.add_nodes_from(sorted_nodes)
    G.add_edges_from(edge_list)

    ### Load Community Info
    membership_list = f'graph_{N}_{mu}.membership'
    membership = np.loadtxt(membership_list, dtype=int)
 
    if disturb_type==4:
        deg_file = f'graph_{N}_{mu}.deg'
        degree = np.loadtxt(deg_file)
        return [G, membership, degree]
    elif disturb_type==5 or disturb_type==6:
        deg_file = f'graph_{N}_{mu}.rank'
        degree = np.loadtxt(deg_file)
        return [G, membership, degree]
    else:
        ### Load Betweeness
        btwn_file = f'graph_{N}_{mu}.between'
        between = np.loadtxt(btwn_file)
        return [G, membership, between]
