import os

import jax.numpy as jnp
import networkx as nx
import numpy as np
from flax.training import checkpoints
from scipy.stats import stats

from generator import DataGenerator
from LOVO_DL_vs_baseline_vs_LiNGAM import random_dag_with_two_arrows, \
    generate_data_from_dag, \
    variable_split

from jax_model.transitive_predictor import TransitivePredictor
from LOVO_DL_vs_baseline_vs_LiNGAM import compute_list_of_arrows


class TrivialGraphDataset:
    def __init__(self,
                 num_datasets: int = 1000,
                 num_nodes=4,
                 num_samples=100,
                 expected_out_degree: int = 1,
                 mechanism: str = 'linear',
                 noise: str = 'uniform'):
        self.num_nodes = num_nodes
        self.num_samples = num_samples
        self.expected_out_degree = expected_out_degree
        self.mechanism = mechanism
        self.noise = noise
        self.cache = []
        self.num_datasets = num_datasets
        self.current = 0

        self.generator = DataGenerator(num_nodes=num_nodes, mechanism=mechanism, graph_type='dag',
                              erdos_p=expected_out_degree / 2 * (num_nodes - 1)
                              )

        for _ in range(self.num_datasets):
            data, ground_truth = self.generator.generate()
            data = np.concatenate([data, np.zeros((num_samples, 1)) - 1], axis=-1)
            adj_matrix = nx.adjacency_matrix(ground_truth).todense()
            self.cache.append((jnp.asarray(data), jnp.asarray(adj_matrix).flatten()))

    def __len__(self):
        return self.num_datasets

    def __getitem__(self, idx):
        return self.cache[idx]

    def __next__(self):
        while self.current < self.num_datasets:
            yield self.cache[self.current]
            self.current += 1


class GraphSplitDatasetGenerator:
    def __init__(self, num_datasets=1000, num_samples=100):
        self.num_datasets = num_datasets
        self.num_samples = num_samples
        self.cache = []
        self.current = 0

        for _ in range(num_datasets):
            permutation, middle_node = random_dag_with_two_arrows()
            XYZ = generate_data_from_dag(permutation, middle_node, num_samples)

            XZtrain, YZtrain, XYZtest = variable_split(XYZ)

            rhoXY = stats.pearsonr(XYZtest[:, 0], XYZtest[:, 1]).statistic
            self.cache.append((jnp.asarray(XZtrain), jnp.asarray(YZtrain), jnp.asarray(rhoXY)))

    def __len__(self):
        return self.num_datasets

    def __getitem__(self, idx):
        return self.cache[idx]

    def __next__(self):
        while self.current < self.num_datasets:
            yield self.cache[self.current]
            self.current += 1

# Deep Learning Model hyperparameters. TODO store these in checkpoint
NUM_ENC_LAYERS = 3
NUM_DEC_LAYERS = 1
NUM_HEADS = 8  # Number of attention heads.
MODEL_SIZE = 64
DROPOUT_RATE = 0.0
LEARNING_RATE = 1e-4
ACC_GRADS = 1

class StructureDatasetGenerator:
    def __init__(self, num_datasets=1000, num_samples=100):
        self.num_datasets = num_datasets
        self.num_samples = num_samples
        self.cache = []
        self.current = 0
        self.model = TransitivePredictor(
            max_num_nodes=2,
            hidden_dim=MODEL_SIZE,
            depth_encoder=NUM_ENC_LAYERS,
            depth_decoder=NUM_DEC_LAYERS,
            dropout_rate=DROPOUT_RATE, num_heads=NUM_HEADS,
            lr=LEARNING_RATE,
            accumulate_grads=ACC_GRADS
            )
        self.dl_model_state = checkpoints.restore_checkpoint(os.path.join('checkpoint_best'), target=None)

        for da in range(num_datasets):
            print(da)
            permutation, middle_node = random_dag_with_two_arrows()
            XYZ = generate_data_from_dag(permutation, middle_node, num_samples)

            XZtrain, _, _ = variable_split(XYZ)
            edges = compute_list_of_arrows(permutation, middle_node)
            if 'X->Y' in edges and 'Z->Y' in edges:
                label = jnp.asarray([1, 0, 0, 0])
            elif 'Y->X' in edges and 'Y->Z' in edges:
                label = jnp.asarray([0, 1, 0, 0])
            elif 'X->Z' in edges or ('X->Y' in edges and 'Y->Z' in edges):
                label = jnp.asarray([0, 0, 1, 0])
            elif 'Z->X' in edges or ('Z->Y' in edges and 'Y->X' in edges):
                label = jnp.asarray([0, 0, 0, 1])
            else:
                print(edges)
                raise NotImplementedError()

            embedding = self.model.apply({'params': self.dl_model_state['params']}, XZtrain, method=self.model.embedd)

            self.cache.append((embedding, label))

    def __len__(self):
        return self.num_datasets

    def __getitem__(self, idx):
        return self.cache[idx]

    def __next__(self):
        while self.current < self.num_datasets:
            yield self.cache[self.current]
            self.current += 1

