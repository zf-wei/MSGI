import networkx as nx
import numpy as np
from networkx.generators.community import LFR_benchmark_graph
from WGE.extract_membership import extract_intrinsic_membership
from WGE.remove_procedure_wp import generate_remove_procedure_wp

n = 1000
tau1 = 2  # Power-law exponent for the degree distribution
tau2 = 1.1  # Power-law exponent for the community size distribution
# mu = 0.1  # Mixing parameter
avg_deg = 25  # Average Degree
max_deg = int(0.1 * n)  # Max Degree
min_commu = 60  # Min Community Size
max_commu = int(0.1 * n)  # Max Community Size

MU = [0.55]

for mu in MU:
    G = LFR_benchmark_graph(
        n, tau1, tau2, mu, average_degree=avg_deg, max_degree=max_deg, min_community=min_commu, max_community=max_commu,
        seed=7
    )

    # Remove multi-edges and self-loops from G
    G = nx.Graph(G)
    selfloop_edges = list(nx.selfloop_edges(G))
    G.remove_edges_from(selfloop_edges)


    nx.write_edgelist(G, f"graph_{n}_{mu}.edgelist", delimiter=' ', data=False)
    
    intrinsic_membership = extract_intrinsic_membership(G)
    membership_output_file = f"graph_{n}_{mu}.membership"
    np.savetxt(membership_output_file, intrinsic_membership, delimiter=' ', fmt='%d')

    # Get betweenness centrality and save it to a file
    betweenness = list(nx.betweenness_centrality(G).values())
    betweenness_output_file = f"graph_{n}_{mu}.between"
    np.savetxt(betweenness_output_file, betweenness, delimiter=' ')
