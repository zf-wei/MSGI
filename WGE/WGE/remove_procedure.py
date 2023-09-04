### 生成删除顶点的顺序 模块
import random
import networkx as nx
import numpy as np
from multiprocessing import Pool
import math
import time


def nodes_sample(disturb_type: int, graph, number_of_nodes: int, percent, betweenness):
    graph_copy = graph.copy()
    sample_size = int(number_of_nodes * percent)
    if disturb_type==1:
        removed_nodes = random.sample(range(number_of_nodes), sample_size)
    elif disturb_type==3: #这段代码需要重写 用无放回抽样
        temp = max(betweenness)
        trans_btwn = [math.tan(x * 0.45 * math.pi / temp) for x in betweenness]
        removed_nodes = random.choices(range(number_of_nodes), trans_btwn, k=sample_size)
    elif disturb_type==6: #这段代码需要重写 用无放回抽样
        trans_rank = [math.exp(x) for x in betweenness]
        removed_nodes = random.choices(range(number_of_nodes), trans_rank, k=sample_size)
    else:
        #######################
        # Generate a new random seed using the current system time
        random_seed = int(time.time())
        # Set the random seed
        np.random.seed(random_seed)
        #######################
        total = sum(betweenness)
        probabilities = [btw / total for btw in betweenness]
        removed_nodes = np.random.choice(range(number_of_nodes), size=sample_size, replace=False, p=probabilities)
        removed_nodes = [int(x) for x in removed_nodes]
        #removed_nodes = random.choices(range(number_of_nodes), betweenness, k=sample_size)
    graph_copy.remove_nodes_from(removed_nodes)
    if nx.is_connected(graph_copy):
        return removed_nodes
        

#import numpy as np
import json

def generate_remove_procedure(disturb_type: int, mu, graph, number_of_nodes, betweenness, sample_count=50):
    remove_procedure = []
    #i=0 
    for percent in np.arange(0.05, 0.86, 0.05):
        ls = []
        while len(ls) < sample_count:
            temp = nodes_sample(disturb_type=disturb_type, graph=graph, number_of_nodes=number_of_nodes, percent=percent, betweenness=betweenness)
            if temp is not None:
                #i=i+1
                #print(i)
                ls.append(temp)
        remove_procedure.append(ls)
    if disturb_type==1:
        filename = f"graph_{graph.number_of_nodes()}_{mu}.stoch_rmv"
    elif disturb_type==2:
        filename = f"graph_{graph.number_of_nodes()}_{mu}.btwn_rmv"
    elif disturb_type==3:
        filename = f"graph_{graph.number_of_nodes()}_{mu}.trans_rmv"
    elif disturb_type==4:
        filename = f"graph_{graph.number_of_nodes()}_{mu}.deg_rmv"
    elif disturb_type==5:
        filename = f"graph_{graph.number_of_nodes()}_{mu}.rank_rmv"
    elif disturb_type==6:
        filename = f"graph_{graph.number_of_nodes()}_{mu}.trank_rmv"
    with open(filename, 'w') as file:
        json.dump(remove_procedure, file)

def call_nodes_sample(args):
    disturb_type, graph, number_of_nodes, percent, betweenness = args
    return nodes_sample(disturb_type=disturb_type, graph=graph, number_of_nodes=number_of_nodes, percent=percent, betweenness=betweenness)
    
def generate_remove_procedure_parallel(disturb_type: int, mu, graph, number_of_nodes, betweenness, upperbound, sample_count=50):
    remove_procedure = []
    
    for percent in np.arange(0.05, upperbound+0.01, 0.05):
        ls = []
        successful_samples = 0
        args_list = [(disturb_type, graph, number_of_nodes, percent, betweenness)] * 128

        while successful_samples < sample_count:
            pool = Pool()
            print("returned")
            results = pool.map(call_nodes_sample, args_list)
            print("processing")
            for temp in results:
                if temp is not None:
                    ls.append(temp)
                    successful_samples += 1
            print(successful_samples)

        remove_procedure.append(ls[:sample_count])
        print(f"{percent}，我是分割线")

    pool.close()
    pool.join()

    if disturb_type==1:
        filename = f"graph_{graph.number_of_nodes()}_{mu}.stoch_rmv"
    elif disturb_type==2:
        filename = f"graph_{graph.number_of_nodes()}_{mu}.btwn_rmv"
    elif disturb_type==3:
        filename = f"graph_{graph.number_of_nodes()}_{mu}.trans_rmv"
    elif disturb_type==4:
        filename = f"graph_{graph.number_of_nodes()}_{mu}.deg_rmv"
    elif disturb_type==5:
        filename = f"graph_{graph.number_of_nodes()}_{mu}.rank_rmv"
    elif disturb_type==6:
        filename = f"graph_{graph.number_of_nodes()}_{mu}.trank_rmv"
    with open(filename, 'w') as file:
        json.dump(remove_procedure, file)
               

def remove_procedure_index(remove_procedure, num_nodes):
    index = []
    for sublist_list in remove_procedure:
        sublist_index = []
        for sublist in sublist_list:
            temp = np.full(num_nodes, True)
            temp[sublist] = False
            sublist_index.append(temp)
        index.append(sublist_index)
    return index
