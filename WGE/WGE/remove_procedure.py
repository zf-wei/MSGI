#from auxpack.utils import generate_output
#################################
#调用 generate_output 函数
#################################

### 生成删除顶点的顺序 模块
import random
import networkx as nx
import numpy as np


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
    i=0 
    for percent in np.arange(0.05, 0.86, 0.05):
        ls = []
        while len(ls) < sample_count:
            temp = nodes_sample(random_disturb=random_disturb, graph=graph, number_of_nodes=number_of_nodes, percent=percent, betweenness=betweenness)
            if temp is not None:
                i=i+1
                print(i)
                ls.append(temp)
        remove_procedure.append(ls)
    filename = f"graph_{graph.number_of_nodes()}_{mu}.rmvproc"
    with open(filename, 'w') as file:
        json.dump(remove_procedure, file)
        

#import numpy as np

def remove_procedure_index(remove_procedure, num_nodes):
    index = []
    for sublist_list in remove_procedure:
        sublist_index = []
        for sublist in sublist_list:
            temp = np.ones(num_nodes, dtype=bool)
            temp[sublist] = False
            sublist_index.append(temp)
        index.append(sublist_index)
    return index