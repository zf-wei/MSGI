import numpy as np
# Get intrinsic membership as an np.array
def extract_intrinsic_membership(G):
    intrinsic_communities = {frozenset(G.nodes[v]["community"]) for v in G}
    intrinsic_membership = np.empty(G.number_of_nodes(), dtype=int)
    for node in range(G.number_of_nodes()):
        for index, inner_set in enumerate(intrinsic_communities):
            if node in inner_set:
                intrinsic_membership[node] = index
                break
    return intrinsic_membership