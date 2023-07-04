import random
import warnings
from typing import List
import re
import numpy as np
import networkx as nx
from gensim.models.word2vec import Word2Vec
from karateclub.utils.walker import RandomWalker


class DeepWalk:
    r"""An implementation of `"DeepWalk" <https://arxiv.org/abs/1403.6652>`_
    from the KDD '14 paper "DeepWalk: Online Learning of Social Representations".
    The procedure uses random walks to approximate the pointwise mutual information
    matrix obtained by pooling normalized adjacency matrix powers. This matrix
    is decomposed by an approximate factorization technique.

    Args:
        walk_number (int): Number of random walks. Default is 10.
        walk_length (int): Length of random walks. Default is 80.
        dimensions (int): Dimensionality of embedding. Default is 128.
        workers (int): Number of cores. Default is 4.
        window_size (int): Matrix power order. Default is 5.
        epochs (int): Number of epochs. Default is 1.
        use_hierarchical_softmax (bool): Whether to use hierarchical softmax or negative sampling to train the model. Default is True.
        number_of_negative_samples (int): Number of negative nodes to sample (usually between 5-20). If set to 0, no negative sampling is used. Default is 5.
        learning_rate (float): HogWild! learning rate. Default is 0.05.
        min_count (int): Minimal count of node occurrences. Default is 1.
        seed (int): Random seed value. Default is 42.
    """

    def __init__(
        self,
        walk_number: int = 10,
        walk_length: int = 80,
        dimensions: int = 128,
        workers: int = 4,
        window_size: int = 5,
        epochs: int = 1,
        use_hierarchical_softmax: bool = True,
        number_of_negative_samples: int = 5,
        learning_rate: float = 0.05,
        min_count: int = 1,
        seed: int = 42,
    ):
        self.walk_number = walk_number
        self.walk_length = walk_length
        self.dimensions = dimensions
        self.workers = workers
        self.window_size = window_size
        self.epochs = epochs
        self.use_hierarchical_softmax = use_hierarchical_softmax
        self.number_of_negative_samples = number_of_negative_samples
        self.learning_rate = learning_rate
        self.min_count = min_count
        self.seed = seed

    def fit(self, graph: nx.classes.graph.Graph):
        """
        Fitting a DeepWalk model.

        Arg types:
            * **graph** *(NetworkX graph)* - The graph to be embedded.
        """
        self._set_seed()
        graph = self._check_graph(graph)
        walker = RandomWalker(self.walk_length, self.walk_number)
        walker.do_walks(graph)

        model = Word2Vec(
            walker.walks,
            hs=1 if self.use_hierarchical_softmax else 0,
            negative=self.number_of_negative_samples,
            alpha=self.learning_rate,
            epochs=self.epochs,
            vector_size=self.dimensions,
            window=self.window_size,
            min_count=self.min_count,
            workers=self.workers,
            seed=self.seed,
        )

        num_of_nodes = graph.number_of_nodes()
        self._embedding = [model.wv[str(n)] for n in list(graph.nodes())]

    def get_embedding(self) -> np.array:
        r"""Getting the node embedding.

        Return types:
            * **embedding** *(Numpy array)* - The embedding of nodes.
        """
        return np.array(self._embedding)

    def get_params(self):
        """Get parameter dictionary for this estimator.."""
        rx = re.compile(r'^\_')
        params = self.__dict__
        params = {key: params[key] for key in params if not rx.search(key)}
        return params

    def _set_seed(self):
        """Creating the initial random seed."""
        random.seed(self.seed)
        np.random.seed(self.seed)

    @staticmethod
    def _ensure_walk_traversal_conditions(graph: nx.classes.graph.Graph) -> nx.classes.graph.Graph:
        """Ensure walk traversal conditions."""
        # We create a copy of the graph
        graph = graph.copy()
        for node_index in list(graph.nodes()):
            if not graph.has_edge(node_index, node_index):
                # And we add the missing edges
                # for filling the main diagonal
                graph.add_edges_from(
                    [(index, index) for index in list(graph.nodes()) if not graph.has_edge(index, index)]
                )
                break

        return graph

    @staticmethod
    def _check_indexing(graph: nx.classes.graph.Graph):
        """Checking the consecutive numeric indexing."""
        numeric_indices = [index for index in range(graph.number_of_nodes())]
        node_indices = sorted([node for node in graph.nodes()])

        assert numeric_indices == node_indices, "The node indexing is wrong."

    def _check_graph(self, graph: nx.classes.graph.Graph) -> nx.classes.graph.Graph:
        """Check the Karate Club assumptions about the graph."""
        #self._check_indexing(graph)
        graph = self._ensure_walk_traversal_conditions(graph)

        return graph

    def _check_graphs(self, graphs: List[nx.classes.graph.Graph]):
        """Check the Karate Club assumptions for a list of graphs."""
        graphs = [self._check_graph(graph) for graph in graphs]

        return graphs