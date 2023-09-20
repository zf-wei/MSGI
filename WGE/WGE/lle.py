###############################
### Correct Graph LLE
###############################
import numpy as np
import networkx as nx
from sklearn.preprocessing import normalize
import scipy.linalg as lg

import cupy as cp

def lle_cupy(graph, dim):
    A = cp.asarray(nx.to_numpy_array(graph, nodelist=graph.nodes(), weight='weight'))
    
    # Manually compute L1 normalization along axis 1 (rows)
    row_sums = cp.sum(A, axis=1)
    A /= row_sums.reshape(-1, 1)

    I_n = cp.eye(graph.number_of_nodes())
    I_min_A = cp.dot((I_n - A).T, (I_n - A))
    w, v = cp.linalg.eigh(I_min_A)
    idx = cp.argsort(w.real)
    v = v[:, idx]
    embedding = v[:, 1:(dim+1)]
    return embedding.get().real  # Explicitly convert to NumPy array using .get()

def lle_np(graph, dim):
    A = nx.to_numpy_array(graph, nodelist=graph.nodes(), weight='weight')
    normalize(A, norm='l1', axis=1, copy=False)
    I_n = np.eye(graph.number_of_nodes())
    I_min_A = np.dot((I_n - A).T, (I_n - A))
    w, v = np.linalg.eig(I_min_A)
    idx = np.argsort(w.real)
    v = v[:, idx]
    embedding = v[:, 1:(dim+1)]
    return embedding.real

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

import scipy.sparse as sp
import scipy.sparse.linalg as splg
def lles(graph, dim):
    A = nx.to_scipy_sparse_matrix(graph, weight='weight')
    A = normalize(A, norm='l1', axis=1)
    I_n = sp.eye(graph.number_of_nodes())
    I_min_A = (I_n - A).T.dot(I_n - A)
    w, v = splg.eigs(I_min_A, k=dim + 1, which='SM')
    idx = np.argsort(w.real)
    v = v[:, idx]
    embedding = v[:, 1:(dim + 1)]
    return embedding.real