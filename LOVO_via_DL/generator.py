from collections import defaultdict
from typing import Tuple

import networkx as nx
import numpy as np
import pandas as pd
from numpy._typing import NDArray


class DataGenerator:

    def __init__(self, num_nodes: int = 8, erdos_p: float = None, mechanism: str = 'cam', graph_type: str = 'dag',
                 coeff_range: Tuple[float, float] = (-1, 1)):
        if graph_type != 'dag':
            raise NotImplementedError('Graph Type can only be "dag" right now. Not ' + str(graph_type))
        self.num_nodes = num_nodes
        self.nodes = ['V{}'.format(i + 1) for i in range(num_nodes)]
        self.p = (1.1 * np.log(num_nodes)) / num_nodes if erdos_p is None else erdos_p
        self.graph = nx.DiGraph()
        self.graph.add_nodes_from(self.nodes)
        self.causal_order = np.random.permutation(self.nodes)
        self.parent_idxs = defaultdict(lambda: [])
        df_idx = {n: i for (i, n) in enumerate(self.nodes)}
        for i, x in enumerate(self.causal_order):
            for j, y in enumerate(self.causal_order):
                if j > i and np.random.rand() < self.p:
                    self.parent_idxs[y].append(df_idx[x])
                    self.graph.add_edge(x, y)

        self.mechanism = {}
        for node in self.parent_idxs.keys():
            if mechanism == 'linear':
                self.mechanism[node] = LinearMechanism(len(self.parent_idxs[node]), coeff_range)
            elif mechanism == 'nn':
                num_parents = len(self.parent_idxs[node])
                self.mechanism[node] = NNMechanism(num_parents, 20, coeff_range=coeff_range)
            elif mechanism == 'cam':
                num_parents = len(self.parent_idxs[node])
                self.mechanism[node] = CAMMechanism(num_parents, 20, coeff_range=coeff_range)
            elif mechanism == 'camsq':
                num_parents = len(self.parent_idxs[node])
                self.mechanism[node] = CAMPolyMechanism(num_parents)
            else:
                raise NotImplementedError('Non-linear model not implemented yet')

    def generate(self, num_samples: int = 100, var: float = 1, noise: str = 'gaussian') -> Tuple[
        pd.DataFrame, nx.DiGraph]:
        sample = pd.DataFrame(np.zeros((num_samples, self.num_nodes)), columns=self.nodes)
        if noise == 'gaussian':
            n_func = lambda: np.random.normal(loc=0, scale=var, size=num_samples)
        elif noise == 'uniform':
            a = np.sqrt(3 * var)  # get var as variance
            n_func = lambda: np.random.uniform(low=-a, high=a, size=num_samples)
        else:
            raise NotImplementedError('Invalid noise parameter: {}'.format(noise))
        for node in self.causal_order:
            values = n_func()  # Right now only additive noise
            if node in self.mechanism:
                values += self.mechanism[node](sample.iloc[:, self.parent_idxs[node]].to_numpy())
            sample.loc[:, node] = values
        return sample, self.graph


class LinearMechanism:
    def __init__(self, num_parents: int, coeff_range: Tuple[float, float], weights: NDArray = None):
        if weights is None:
            self.weights = np.random.uniform(low=coeff_range[0], high=coeff_range[1], size=num_parents) \
                           * np.random.choice([-1, 1])
        else:
            self.weights = weights

    def __call__(self, parents: NDArray) -> NDArray:
        return np.dot(self.weights, parents.T).T


class NNMechanism:
    def __init__(self, num_parents: int, num_hidden: int = 10, coeff_range: Tuple[float, float] = (-1, 1)):
        self.weights_in = np.random.uniform(low=coeff_range[0], high=coeff_range[1], size=(num_hidden, num_parents))
        self.bias = np.random.uniform(low=coeff_range[0], high=coeff_range[1])
        self.weights_out = np.random.uniform(low=-coeff_range[0], high=coeff_range[1], size=num_hidden)

    def __call__(self, parents: NDArray) -> NDArray:
        hidden = np.dot(self.weights_in, parents.T) + self.bias
        transformed = np.tanh(hidden)
        return np.dot(self.weights_out, transformed).T


class CAMMechanism:
    def __init__(self, num_parents: int, num_hidden: int = 10, coeff_range: Tuple[float, float] = (-1, 1)):
        self.mechanisms = []
        for _ in range(num_parents):
            self.mechanisms.append(NNMechanism(1, num_hidden, coeff_range))

    def __call__(self, parents: NDArray) -> NDArray:
        output = np.zeros(parents.shape[0])
        for i in range(parents.shape[1]):
            output += self.mechanisms[i](np.expand_dims(parents[:, i], -1))
        return output


class CAMPolyMechanism:
    def __init__(self, num_parents: int, max_degree: int = 5, coeff_range: Tuple[float, float] = (-1, 1)):
        self.mechanisms = []
        for _ in range(num_parents):
            coefs = np.random.uniform(low=coeff_range[0], high=coeff_range[1], size=max_degree)
            self.mechanisms.append(lambda pa: np.sum([coefs[i] * pa ** i for i in range(max_degree)], axis=0)[:, 0])

    def __call__(self, parents: NDArray) -> NDArray:
        output = np.zeros(parents.shape[0])
        for i in range(parents.shape[1]):
            output += self.mechanisms[i](np.expand_dims(parents[:, i], -1))
        return output
