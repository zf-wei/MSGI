### 生成删除顶点的顺序 模块
import random
import networkx as nx
import numpy as np
from multiprocessing import Pool


def nodes_sample(random_disturb: bool, graph, number_of_nodes: int, percent, betweenness):
    graph_copy = graph.copy()
    sample_size = int(number_of_nodes * percent)
    if random_disturb:
        removed_nodes = random.sample(range(number_of_nodes), sample_size)
    else: 
        removed_nodes = random.choices(range(number_of_nodes), betweenness, k=sample_size)
    graph_copy.remove_nodes_from(removed_nodes)
    if nx.is_connected(graph_copy):
        return removed_nodes
        

#import numpy as np
import json

def generate_remove_procedure(random_disturb: bool, mu, graph, number_of_nodes, betweenness, sample_count=50):
    remove_procedure = []
    #i=0 
    for percent in np.arange(0.05, 0.86, 0.05):
        ls = []
        while len(ls) < sample_count:
            temp = nodes_sample(random_disturb=random_disturb, graph=graph, number_of_nodes=number_of_nodes, percent=percent, betweenness=betweenness)
            if temp is not None:
                #i=i+1
                #print(i)
                ls.append(temp)
        remove_procedure.append(ls)
    if random_disturb:
        filename = f"graph_{graph.number_of_nodes()}_{mu}.stoch_rmv"
    else:
        filename = f"graph_{graph.number_of_nodes()}_{mu}.btwn_rmv"
    with open(filename, 'w') as file:
        json.dump(remove_procedure, file)

def call_nodes_sample(args):
    random_disturb, graph, number_of_nodes, percent, betweenness = args
    return nodes_sample(random_disturb=random_disturb, graph=graph, number_of_nodes=number_of_nodes, percent=percent, betweenness=betweenness)
    
def generate_remove_procedure_parallel(random_disturb: bool, mu, graph, number_of_nodes, betweenness, upperbound, sample_count=50):
    remove_procedure = []
    
    for percent in np.arange(0.05, upperbound+0.01, 0.05):
        ls = []
        successful_samples = 0
        args_list = [(random_disturb, graph, number_of_nodes, percent, betweenness)] * 128

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

    if random_disturb:
        filename = f"graph_{graph.number_of_nodes()}_{mu}.stoch_rmv"
    else:
        filename = f"graph_{graph.number_of_nodes()}_{mu}.btwn_rmv"
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
