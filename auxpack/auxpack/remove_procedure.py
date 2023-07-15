from auxpack.utils import generate_output
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
        graph_copy.remove_nodes_from(removed_nodes)
        if nx.is_connected(graph_copy):
            return removed_nodes
    else: 
        removed_nodes = random.choices(range(number_of_nodes), betweenness, k=sample_size)
        graph_copy.remove_nodes_from(removed_nodes)
        if nx.is_connected(graph_copy):
            return removed_nodes
        

#import numpy as np
import json

def generate_remove_list(random_disturb: bool, graph, number_of_nodes, betweenness):
    remove_procedure = []
    for percent in np.arange(0.05, 0.86, 0.05):
        ls = []
        while len(ls) < 10:
            temp = nodes_sample(random_disturb=random_disturb, graph=graph, number_of_nodes=number_of_nodes, percent=percent, betweenness=betweenness)
            if temp is not None:
                ls.append(temp)
        remove_procedure.append(ls)
    filename = generate_output(random_disturb, "0Remove_Procedure.txt")
    with open(filename, 'w') as file:
        json.dump(remove_procedure, file)
    return remove_procedure


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