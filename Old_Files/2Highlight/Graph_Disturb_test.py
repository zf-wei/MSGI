import argparse
from WGE.graph_input import graph_input
from WGE.processing import Comprehensive_Processing
from WGE.plot import Plot_Total
import multiprocessing

# 创建参数解析器
parser = argparse.ArgumentParser(description='This program performs graph embedding, clustering.')

# 添加命令行参数
parser.add_argument('-N', '--N_value', type=int, help='Number of Vertices')
parser.add_argument('-D', '--D_value', type=int, help='Embedding Dimension')
parser.add_argument('-M', '--M_value', type=int, help='Embedding Method ID')
parser.add_argument('-d', '--disturb_type', type=int, help='Disturb Type?')

# 解析命令行参数
args = parser.parse_args()

# 读取命令行参数的值
N = args.N_value
D = args.D_value
method = args.M_value
disturb_type = args.disturb_type

output_flag = True

if method==7:
    num_cpus = 3
else:
    num_cpus = multiprocessing.cpu_count()

MU = [0.1]


graphs, membership, between, remove_procedure, index = graph_input(N=N, MU=MU, disturb_type=disturb_type)

MEAN = {}
STD = {}
for mu in MU:
    MEAN[mu], STD[mu] = Comprehensive_Processing(output=output_flag, disturb_type=disturb_type, method=method, num_cpus=num_cpus, 
                                                 graph=graphs[mu], embedding_dimension=D, intrinsic_membership=membership[mu], 
                                                 remove_procedure=remove_procedure[mu], remove_procedure_index_form=index[mu], mu=mu)

