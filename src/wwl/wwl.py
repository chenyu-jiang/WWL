# -----------------------------------------------------------------------------
# This file contains the API for the WWL kernel computations
#
# December 2019, M. Togninalli
# -----------------------------------------------------------------------------
from .propagation_scheme import WeisfeilerLehman, ContinuousWeisfeilerLehman
from sklearn.metrics.pairwise import laplacian_kernel
import ot
import numpy as np


def _compute_wasserstein_distance(label_sequences, sinkhorn=False, 
                                    categorical=False, sinkhorn_lambda=1e-2):
    '''
    Generate the Wasserstein distance matrix for the graphs embedded 
    in label_sequences
    '''
    # Get the iteration number from the embedding file
    n = len(label_sequences)
    
    M = np.zeros((n,n))
    # Iterate over pairs of graphs
    for graph_index_1, graph_1 in enumerate(label_sequences):
        # Only keep the embeddings for the first h iterations
        labels_1 = label_sequences[graph_index_1]
        for graph_index_2, graph_2 in enumerate(label_sequences[graph_index_1:]):
            labels_2 = label_sequences[graph_index_2 + graph_index_1]
            # Get cost matrix
            ground_distance = 'hamming' if categorical else 'euclidean'
            costs = ot.dist(labels_1, labels_2, metric=ground_distance)

            if sinkhorn:
                mat = ot.sinkhorn(np.ones(len(labels_1))/len(labels_1), 
                                    np.ones(len(labels_2))/len(labels_2), costs, sinkhorn_lambda, 
                                    numItermax=50)
                M[graph_index_1, graph_index_2 + graph_index_1] = np.sum(np.multiply(mat, costs))
            else:
                M[graph_index_1, graph_index_2 + graph_index_1] = \
                    ot.emd2([], [], costs)
                    
    M = (M + M.T)
    return M

def pairwise_wasserstein_distance(X, node_features = None, num_iterations=3, sinkhorn=False, enforce_continuous=False):
    """
    Pairwise computation of the Wasserstein distance between embeddings of the 
    graphs in X.
    args:
        X (List[ig.graphs]): List of graphs
        node_features (array): Array containing the node features for continuously attributed graphs
        num_iterations (int): Number of iterations for the propagation scheme
        sinkhorn (bool): Indicates whether sinkhorn approximation should be used
    """
    # First check if the graphs are continuous vs categorical
    categorical = True
    if enforce_continuous:
        print('Enforce continous flag is on, using CONTINUOUS propagation scheme.')
        categorical = False
    elif node_features is not None:
        print('Continuous node features provided, using CONTINUOUS propagation scheme.')
        categorical = False
    else:
        for g in X:
            if not 'label' in g.vs.attribute_names():
                print('No label attributed to graphs, use degree instead and use CONTINUOUS propagation scheme.')
                categorical = False
                break
        if categorical:
            print('Categorically-labelled graphs, using CATEGORICAL propagation scheme.')
    
    # Embed the nodes
    if categorical:
        es = WeisfeilerLehman()
        node_representations = es.fit_transform(X, num_iterations=num_iterations)
    else:
        es = ContinuousWeisfeilerLehman()
        node_representations = es.fit_transform(X, node_features=node_features, num_iterations=num_iterations)

    # Compute the Wasserstein distance
    pairwise_distances = _compute_wasserstein_distance(node_representations, sinkhorn=sinkhorn, 
                                    categorical=categorical, sinkhorn_lambda=1e-2)
    return pairwise_distances

def wwl(X, node_features=None, num_iterations=3, sinkhorn=False, gamma=None):
    """
    Pairwise computation of the Wasserstein Weisfeiler-Lehman kernel for graphs in X.
    """
    D_W =  pairwise_wasserstein_distance(X, node_features = node_features, 
                                num_iterations=num_iterations, sinkhorn=sinkhorn)
    wwl = laplacian_kernel(D_W, gamma=gamma)
    return wwl

#######################
# Class implementation
#######################

class PairwiseWWL():
    def __init__(self, X, node_features=None, enforce_continuous=False, num_iterations=3, sinkhorn=False):
        self.num_iterations = num_iterations
        self.sinkhorn = sinkhorn
        self.enforce_continuous = enforce_continuous
        self.node_features = node_features
        self.X = X
        self._distance_cache = {}
        self._compute_node_representation()

    def _compute_node_representation(self):
        # First check if the graphs are continuous vs categorical
        self.categorical = True
        if self.enforce_continuous:
            print('Enforce continous flag is on, using CONTINUOUS propagation scheme.')
            self.categorical = False
        elif self.node_features is not None:
            print('Continuous node features provided, using CONTINUOUS propagation scheme.')
            self.categorical = False
        else:
            for g in self.X:
                if not 'label' in g.vs.attribute_names():
                    print('No label attributed to graphs, use degree instead and use CONTINUOUS propagation scheme.')
                    self.categorical = False
                    break
            if self.categorical:
                print('Categorically-labelled graphs, using CATEGORICAL propagation scheme.')
        
        # Embed the nodes
        if self.categorical:
            es = WeisfeilerLehman()
            node_representations = es.fit_transform(self.X, num_iterations=self.num_iterations)
        else:
            es = ContinuousWeisfeilerLehman()
            node_representations = es.fit_transform(self.X, node_features=self.node_features, num_iterations=self.num_iterations)
        self.node_representations = node_representations

    def wwl_distance(self, idx_1, idx_2, sinkhorn_lambda=1e-2):
        # make idx_1 <= idx_2
        if idx_1 > idx_2:
            idx_1, idx_2 = idx_2, idx_1
        if (idx_1, idx_2) in self._distance_cache:
            return self._distance_cache[(idx_1, idx_2)]

        labels_1 = self.node_representations[idx_1]
        labels_2 = self.node_representations[idx_2]
        # Get cost matrix
        ground_distance = 'hamming' if self.categorical else 'euclidean'
        costs = ot.dist(labels_1, labels_2, metric=ground_distance)

        if self.sinkhorn:
            mat = ot.sinkhorn(np.ones(len(labels_1))/len(labels_1), 
                                np.ones(len(labels_2))/len(labels_2), costs, sinkhorn_lambda, 
                                numItermax=50)
            distance = np.sum(np.multiply(mat, costs))
        else:
            distance = ot.emd2([], [], costs)
        self._distance_cache[(idx_1, idx_2)] = distance
        return distance
    
    def __getitem__(self, indices):
        idx_1, idx_2 = indices
        return self.wwl_distance(idx_1, idx_2)

    @property
    def shape(self):
        return (self.node_representations.shape[0], self.node_representations.shape[0])