###############################
### Correct Graph LLE
###############################
import numpy as np
import networkx as nx
from sklearn.preprocessing import normalize
import scipy.linalg as lg


def lle(graph, dim):
    #A = nx.to_numpy_array(graph, nodelist=sorted(graph.nodes()), weight='weight')
    A = nx.to_numpy_array(graph, nodelist=graph.nodes(), weight='weight')
    normalize(A, norm='l1', axis=1, copy=False)
    I_n = np.eye(graph.number_of_nodes())
    I_min_A = np.dot((I_n - A).T, (I_n - A))
    w, v = lg.eig(I_min_A)
    idx = np.argsort(w.real)
    v = v[:, idx]
    embedding = v[:, 1:(dim+1)]
    return embedding