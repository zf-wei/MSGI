import argparse
from WGE.graph_input import graph_input_simple
import multiprocessing
from WGE.remove_procedure import generate_remove_procedure_parallel

# 创建参数解析器
parser = argparse.ArgumentParser(description='This program performs graph embedding, clustering. Plot will be made accordingly')

# 添加命令行参数
parser.add_argument('-N', '--N_value', type=int, help='Number of Vertices')
parser.add_argument('-r', '--random_disturb', action='store_false', help='Disturb Graph Randomly?')
parser.add_argument('-m', '--mu_value', type=float, help='mu_value')  
parser.add_argument('-u', '--percent_limit', type=float, help='percent_limit')  

# 解析命令行参数
args = parser.parse_args()

# 读取命令行参数的值
N = args.N_value
random_disturb = args.random_disturb
mu = args.mu_value  # Modified this line
upperbound = args.percent_limit


graph, membership, between = graph_input_simple(N=N, mu=mu, random_disturb=random_disturb)  

if random_disturb:
    between = [0] * graph.number_of_nodes()

print(N, mu)
generate_remove_procedure_parallel(random_disturb=random_disturb, mu=mu, graph=graph,
                                   number_of_nodes=graph.number_of_nodes(), betweenness=between, upperbound=upperbound,
                                   sample_count=50)
