import numpy as np
import pandas as pd
import random
import time
import networkx as nx

def simulate_data(nr_nodes, sample_size, dir_edge_density=0.3, bidir_edge_density=0.3, tree = False, linear = True):
    """
    Simulates data Markkov to a graph G for a given number of nodes and sample size with specified edge densities.
    Parameters:
    nr_nodes (int): Number of nodes in the graph.
    sample_size (int): Number of samples to generate.
    dir_edge_density (float, optional): Density of directed edges in the graph. Default is 0.3.
    bidir_edge_density (float, optional): Density of bidirected edges in the graph. Default is 0.3.
    tree (bool, optional): If True, the graph is a tree. Default is False.
    linear (bool, optional): If True, the data is generated using a linear SEM, otherwise using a post non-linear model. Default is True.

    Returns:
    pd.DataFrame: The simulated data.
    np.ndarray: The adjacency matrix of G. Bidirected edges are encoded by adding explicit latent variables, which correspond to the columns
    p+1, ..., last column of the adjacency matrix.
    list: The topological order.
    """
    random.seed(time.time())
    if not linear and bidir_edge_density != 0:
        raise NotImplementedError("Combination of non-linear SEM and bidirected edges not implemented.")

    if tree:
        tree = nx.random_tree(nr_nodes)
        dir_edges_present = nx.adjacency_matrix(tree).todense()
        # nx tree has undirected edges i - j, encoded by a 1 in both (i,j) and (j,i). To get bidirected edges, we remove the upper triangle
        dir_edges_present = np.tril(dir_edges_present, -1)
    else:
        dir_edges_present = np.tril(np.random.choice([0, 1], size=(nr_nodes, nr_nodes), p=[1-dir_edge_density, dir_edge_density]), k=-1)
    
    dir_edge_strengths = np.random.uniform(0.5, 1, (nr_nodes, nr_nodes)) * np.random.choice([-1, 1], size=(nr_nodes, nr_nodes))
    Lambda = dir_edges_present * dir_edge_strengths

    # Random order
    permutation = np.random.permutation(nr_nodes)
    Lambda = Lambda[:, permutation][permutation, :]
    topological_order = list(np.argsort(permutation))

    bidir_edges_present = np.tril(np.random.choice([0, 1], size=(nr_nodes, nr_nodes), p=[1-bidir_edge_density, bidir_edge_density]), k=-1)
    # Add explicit latent for each bidirected edge
    bidir_edges = list(np.argwhere(bidir_edges_present != 0))
    Gamma = np.zeros((nr_nodes, len(bidir_edges)))
    for i, bidir_edge in enumerate(bidir_edges):
        u, v = bidir_edge[0], bidir_edge[1]
        Gamma[u, i] = 1
        Gamma[v, i] = np.random.uniform(0.5, 1) * np.random.choice([-1, 1])

    noise = np.random.uniform(-1,1, (sample_size, nr_nodes + Gamma.shape[1]))
    if linear:
        mixing_matrix = np.linalg.inv(np.eye(nr_nodes)-Lambda) @ np.hstack((np.eye(nr_nodes), Gamma))
        data = noise @ mixing_matrix.T
    else:
        data = np.zeros((sample_size, nr_nodes))
        for i in topological_order:
            parents = np.argwhere(Lambda[i] != 0).flatten()
            if len(parents) == 0:
                data[:, i] = noise[:, i]
            else:
                data[:, i] = np.vectorize(lambda x: x**2 - x)(data[:, parents] @ Lambda[i, parents] + noise[:, i])

    column_names = [j for j in range(nr_nodes)]
    return pd.DataFrame(data, columns= column_names), np.hstack((Lambda, Gamma)), topological_order
