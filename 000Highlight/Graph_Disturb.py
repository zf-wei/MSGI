import argparse
from WGE.graph_input import graph_input
from WGE.processing import Comprehensive_Processing
from WGE.plot import Plot_Total
import multiprocessing

# 创建参数解析器
parser = argparse.ArgumentParser(description='This program performs graph embedding, clustering. Plot will be made accordingly')

# 添加命令行参数
parser.add_argument('-N', '--N_value', type=int, help='Number of Vertices')
parser.add_argument('-D', '--D_value', type=int, help='Embedding Dimension')
parser.add_argument('-M', '--M_value', type=int, help='Embedding Method ID')
parser.add_argument('-r', '--random_disturb', help='Disturb Graph Randomly?')

# 解析命令行参数
args = parser.parse_args()

# 读取命令行参数的值
N = args.N_value
D = args.D_value
method = args.M_value
random_disturb = args.random_disturb

output_flag = True
num_cpus_n2v = 2

num_cpus = multiprocessing.cpu_count()

MU = [0.015, 0.1, 0.2, 0.3, 0.4, 0.5]


graphs, membership, between, remove_procedure, index = graph_input(N=N, MU=MU, random_disturb=random_disturb)

MEAN = {}
STD = {}
for mu in MU:
    MEAN[mu], STD[mu] = Comprehensive_Processing(output=output_flag, random_disturb=random_disturb, method=method, num_cpus=num_cpus, 
                                                 graph=graphs[mu], embedding_dimension=D, intrinsic_membership=membership[mu], 
                                                 remove_procedure=remove_procedure[mu], remove_procedure_index_form=index[mu], mu=mu)

measurements = ["NMI", "NMI", "ECSim", "ECSim"]
tokens = ["Euclidean", "Spherical", "Euclidean", "Spherical"]

labels = ["1HOPE", "2LAP", "3LLE", "4DeepWalk", "5MNMF", "6LINE", "7Node2Vec"]

for tok in range(4):
    MEAN_Organized = []
    STD_Organized = []

    for i in range(18):
        mm = [mean[i][tok] for mean in MEAN.values()]
        MEAN_Organized.append(mm)

        ss = [std[i][tok] for std in STD.values()]
        STD_Organized.append(ss)

    measurement = measurements[tok]
    token = tokens[tok]

    filename = f"{N}_{D}-dim_{labels[method-1]}_{token}_{measurement}"

        
    Plot_Total(output=output_flag, random_disturb=random_disturb, measurement=measurement, 
               title=f"{D}-dimension, {token}", MEAN=MEAN_Organized, STD=STD_Organized, filename=filename)
