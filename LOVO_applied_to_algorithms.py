import numpy as np
import itertools as it
import lingam
from dodiscover import PC, FCI, make_context
from dodiscover.ci import FisherZCITest, KernelCITest, GSquareCITest, CMITest, ClassifierCITest, ClassifierCMITest
from joblib import Parallel, delayed
from tqdm import tqdm
import os
import sys
import logging
import networkx as nx
from simulate_data import simulate_data
from extended_pywhy_graphs import ExtendedADMG, calculate_shd, ExtendedPAG
from LOVO_via_parent_adjustment import lovo_comparison

from pandarallel import pandarallel
import pandas as pd
import random

import logging
logger = logging.getLogger()
logger.setLevel(logging.WARNING)

nr_runs = 100
n_val = 5000
n_learns = [100, 500, 1000, 5000]

def apply_algorithm(algorithm, data: pd.DataFrame, leave_out: int):
    logger = logging.getLogger()
    logger.setLevel(logging.WARNING)    
    data = data.drop(columns=leave_out)
    
    if algorithm.__class__.__name__ in ['PC', 'FCI']:
        algorithm.learn_graph(data, make_context().variables(data=data).build())
        if algorithm.__class__.__name__ == 'FCI':
            try:
                learned_pag = algorithm.graph_
                return ExtendedPAG(incoming_nodes=data.columns,
                                incoming_bidirected_edges=learned_pag.bidirected_edges,
                                incoming_directed_edges=learned_pag.directed_edges,
                                incoming_undirected_edges=learned_pag.undirected_edges,
                                incoming_circle_edges=learned_pag.circle_edges)
            except RuntimeError:
                return None
        else:
            return algorithm.graph_
    else: 
        algorithm.fit(data.to_numpy())
        adj_matrix = algorithm.adjacency_matrix_
        # Shift indices again to match the original data
        adj_matrix_expanded = np.insert(adj_matrix, leave_out, 0, axis=0)  
        adj_matrix_expanded = np.insert(adj_matrix_expanded, leave_out, 0, axis=1)
        try:
            result = ExtendedADMG.from_adjacency_matrix(adj_matrix_expanded, RCD_matrix=isinstance(algorithm, lingam.RCD))    
            result.remove_node(leave_out)
        except:
            result = None
        return result

def compute_CV_errors(index, algorithm, assumptions_lemma, nr_nodes, p, n_learn):
    if index % 10 == 0:
        print(f'Repetition {index}')
    np.random.seed(random.randint(0,2**32 - 1))
    data, coeff_matrix, _ = simulate_data(nr_nodes, n_learn + n_val, p, bidir_edge_density=0)
    data_learn, data_val = data[:n_learn], data[n_learn:]
    data_learn = (data_learn - data_learn.mean(axis=0)) / data_learn.std(axis=0)
    
    true_graph = ExtendedADMG.from_adjacency_matrix(coeff_matrix)
    true_graphs_without_i = {i: true_graph.project_to_GX(i) for i in range(nr_nodes)}
    learned_graphs_without_i = {i: apply_algorithm(algorithm, data_learn, leave_out=i) for i in range(nr_nodes)} 
    
    feasible_nodes = [i for i in range(nr_nodes) if learned_graphs_without_i[i] is not None]
    
    if isinstance(algorithm, lingam.RCD):
        # RCD does not learn confounded causal links (compare Section 5.2 in the paper). To enable a comparison, we replace each confounded 
        # causal link in the true graph by a directed edge.
        SHDs = {i: calculate_shd(learned_graphs_without_i[i], true_graphs_without_i[i].to_RCD_ADMG()) for i in feasible_nodes}
    else:
        SHDs = {i: calculate_shd(learned_graphs_without_i[i], true_graphs_without_i[i]) for i in feasible_nodes}
    
    all_pairs = list(it.combinations(feasible_nodes,2))
    lovo_errors, baseline_errors, adjustment_set = zip(*(lovo_comparison(X, Y, 
                                                        learned_graphs_without_i[Y], 
                                                        learned_graphs_without_i[X], 
                                                        data_val.copy(), assumptions_lemma)
                                            for (X, Y) in all_pairs))
    SHDs = [np.sum([SHDs[X], SHDs[Y]]) for X, Y in all_pairs]
    edge_exists = [X in true_graph.neighbors(Y) for X, Y in all_pairs]
    
    return pd.DataFrame({'edge': all_pairs,  
                        'lovo_error': lovo_errors,
                        'baseline_error': baseline_errors,
                        'adjustment_set': adjustment_set,
                        'SHD': SHDs,
                        'edge_exists': edge_exists,
                        'simulation': [index]*len(all_pairs)})

all_results = []
settings = [
            {'algorithm': lingam.DirectLiNGAM(), 'nr_nodes': 10, 'p': 0.5, 'assumptions_lemma': 'DAG'},
            {'algorithm': lingam.RCD(), 'nr_nodes': 5, 'p': 0.3, 'assumptions_lemma': 'GX_ADMG_G_DAG'},
            {'algorithm': PC(CMITest()), 'nr_nodes': 10, 'p': 0.3, 'assumptions_lemma': 'CPDAG'},
            {'algorithm': FCI(CMITest()), 'nr_nodes': 10, 'p': 0.5, 'assumptions_lemma': 'PAG'},
        ]

for setting in settings: 
    algorithm, nr_nodes, p, assumptions_lemma = setting['algorithm'], setting['nr_nodes'], setting['p'], setting['assumptions_lemma']
    print(f'Algorithm: {algorithm.__class__.__name__}')
    num_cores = os.cpu_count()
    for n_learn in n_learns:
        print(f'Number of samples: {n_learn}')
        results = Parallel(n_jobs=min(num_cores-3, 10))(delayed(compute_CV_errors)(i, algorithm, assumptions_lemma, nr_nodes, p, n_learn) for i in tqdm(range(nr_runs)))
        results = pd.concat(results, ignore_index=True)
        results['n_learn'] = n_learn
        all_results.append(results)

all_results = pd.concat(all_results, ignore_index=True)
all_results.to_csv(f"simulation_results/{algorithm.__class__.__name__}_nr_nodes={nr_nodes}_p={p}_reps={nr_runs}_varying_nlearn.csv", index=False) 
ß