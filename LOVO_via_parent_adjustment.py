import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from joblib import Parallel, delayed
import math
import random
import itertools as it
from extended_pywhy_graphs import ExtendedADMG, ExtendedPAG, combine_graphs, find_separating_set_PAG
from pywhy_graphs.networkx.algorithms.causal import m_separated
from pywhy_graphs.viz import draw
from simulate_data import simulate_data
import logging

def edge_might_exist(X, Y, graph_with_X, graph_with_Y, graph_type):
    '''
    Checks if edge between X and Y might exist, based on Lemma 2, 3, or TODO.
    '''
    assert graph_type in ['ADMG', 'DAG', 'GX_ADMG_G_DAG', 'PAG', 'CPDAG']

    if graph_type in ['CPDAG', 'DAG']: 
        if any(c not in graph_with_Y.neighbors(Y) for c in graph_with_X.children(X)):
            return False
        if any(c not in graph_with_X.neighbors(X) for c in graph_with_Y.children(Y)):
            return False
        
        return True

    if graph_type in ['ADMG', 'PAG']:
        ch_X = graph_with_X.children(X)
        ch_Y = graph_with_Y.children(Y)

        if graph_type == 'ADMG':
            possible_sib_X = set(graph_with_X.siblings(X))
            possible_sib_Y = set(graph_with_Y.siblings(Y))
            possible_ch_X = set(graph_with_X.children(X))
            possible_ch_Y = set(graph_with_Y.children(Y))

        if graph_type == 'PAG':
            possible_sib_X = set(graph_with_X.possible_siblings(X))
            possible_sib_Y = set(graph_with_Y.possible_siblings(Y))
            possible_ch_X = set(graph_with_X.possible_children(X))
            possible_ch_Y = set(graph_with_Y.possible_children(Y))

        if any(c not in possible_ch_Y | possible_sib_Y for c in ch_X):
            return False
        if any(c not in possible_ch_X | possible_sib_X for c in ch_Y):
            return False
        
        return True
    
    if graph_type == 'GX_ADMG_G_DAG':
        ch_X = set(graph_with_X.children(X))
        ch_Y = set(graph_with_Y.children(Y))

        X_multiple_children = len(graph_with_Y.bidirected_edges) > 0
        Y_multiple_children = len(graph_with_X.bidirected_edges) > 0
        confounded_nodes_G_X = [u for e in graph_with_X.bidirected_edges for u in e]
        confounded_nodes_G_Y = [u for e in graph_with_Y.bidirected_edges for u in e]
        if X_multiple_children and Y_multiple_children and (X in confounded_nodes_G_X or Y in confounded_nodes_G_Y):
            return True
        if X_multiple_children and not Y_multiple_children and (Y in confounded_nodes_G_Y or len(ch_Y) > 1):
            return True
        if Y_multiple_children and not X_multiple_children and (X in confounded_nodes_G_X or len(ch_X) > 1):
            return True
        if not X_multiple_children and not Y_multiple_children:
            if ch_X == ch_Y:
                pa_X = set(graph_with_X.parents(X))
                pa_Y = set(graph_with_Y.parents(Y))
                if len(ch_X) == 1:
                    C = list(ch_X)[0]
                    pa_C_GX = set(graph_with_X.parents(C))
                    pa_C_GY = set(graph_with_Y.parents(C))
                    if pa_X.issubset(pa_Y) and pa_Y.issubset(pa_X | pa_C_GX):
                        return True
                    elif pa_Y.issubset(pa_X) and pa_X.issubset(pa_Y | pa_C_GY):
                        return True
                else:
                    if pa_X.issubset(pa_Y) or pa_Y.issubset(pa_X):
                        return True
        return False
    
def three_step_LOVO_predictor(X, Y, data_XZ, data_YZ, adjustment_set):
    '''
    Three-step LOVO predictor as defined in Section 3.2.
    '''
    if not adjustment_set:
        return 0
    else:
        data_Y = data_YZ[Y].to_numpy().reshape((-1, 1))
        reg = LinearRegression().fit(data_YZ[adjustment_set].to_numpy(), data_Y)
        Y_pred = np.matmul(data_XZ[adjustment_set], reg.coef_.transpose())[0]
        data_X = data_XZ[X].to_numpy().reshape(-1, 1)
        reg = LinearRegression().fit(data_X, Y_pred)
        return reg.coef_[0]

def lovo_comparison(X, Y, G_X, G_Y, data, graph_type):
    '''
    Estimates the prediction error of the parent adjustment LOVO predictor, along with the error of the baseline predictor with random 
    adjustment sets of equal size.    

    Parameters:
    X (int): First variable.
    Y (int): Second variable.
    G_X (Graph): The marginal graph containing variable X.
    G_Y (Graph): The marginal graph containing variable Y.
    data (pd.DataFrame): The dataset containing the variables.
    graph_type (str): The graph type, relevant for logic to exclude edges. 
    
    Returns:
    tuple: A tuple containing:
        - lovo_error (float): The error of the LOVO predictor.
        - mean_error_random (float): The mean error for random adjustment sets.
        - adjustment_set (list): The adjustment set used by the LOVO predictor.
    '''
    sample_size = data.shape[0]

    size_train = math.floor(sample_size/3)
    data_XZ = data.drop([Y], axis=1).iloc[:size_train,:]
    data_YZ = data.drop([X], axis=1).iloc[size_train:2*size_train,:]
    XY_test = data[[X, Y]].iloc[2*size_train:,:]

    data_XZ = (data_XZ - data_XZ.mean(axis=0)) / data_XZ.std(axis=0)
    data_YZ = (data_YZ - data_YZ.mean(axis=0)) / data_YZ.std(axis=0)
    XY_test = (XY_test - XY_test.mean(axis=0)) / XY_test.std(axis=0)

    if edge_might_exist(X, Y, G_X, G_Y, graph_type):
        return np.nan, np.nan, []
    if graph_type == 'ADMG' and (len(set(G_X.parents(X)) & set(G_X.siblings(X))) > 0 or len(set(G_Y.parents(Y)) & set(G_Y.siblings(Y))) > 0):
        return np.nan, np.nan, []
    if graph_type in ['PAG', 'CPDAG']:
        try:
            G_combined = combine_graphs(G_X, G_Y)
            adjustment_set = find_separating_set_PAG(X, Y, G_combined)
        except:
            print("Combining graphs failed")
            adjustment_set = None
        if adjustment_set is None:
            return np.nan, np.nan, []
    else:
        adjustment_set = list((set(G_X.parents(X)) | set(G_Y.parents(Y))) - {X, Y})
    
        
    all_Z_variables = list(data.drop([X, Y], axis=1).columns)
    true_coeff = np.corrcoef(XY_test, rowvar=False)[0,1]

    lovo_error = abs(three_step_LOVO_predictor(X, Y, data_XZ, data_YZ, adjustment_set) - true_coeff)

    mean_error_random = 0
    number_adjustment_sets = 100
    if len(adjustment_set) != 0:
        for _ in range(number_adjustment_sets):
            random_adjustment_set = random.sample(all_Z_variables, len(adjustment_set))
            mean_error_random += abs(three_step_LOVO_predictor(X, Y, data_XZ, data_YZ, random_adjustment_set) - true_coeff)
        mean_error_random = mean_error_random/number_adjustment_sets
    else:
        mean_error_random =  abs(three_step_LOVO_predictor(X, Y, data_XZ, data_YZ, all_Z_variables) - true_coeff)

    return lovo_error, mean_error_random, adjustment_set 

def one_repetition(i, nr_nodes, sample_size, p, q, lemma):
    logger = logging.getLogger()
    logger.setLevel(logging.WARNING)    
    if i % 50 == 0:
        print(i)
    data, coeff_matrix, _ = simulate_data(nr_nodes, sample_size, p, bidir_edge_density=q)
    true_graph = ExtendedADMG.from_adjacency_matrix(coeff_matrix)
    if lemma == 'CPDAG':
        graphs_without_node_i = {i: true_graph.project_to_GX(i).to_CPDAG() for i in range(nr_nodes)}
    elif lemma == 'PAG':
        graphs_without_node_i = {i: true_graph.project_to_GX(i).to_PAG() for i in range(nr_nodes)}
    else:
        graphs_without_node_i = {i: true_graph.project_to_GX(i) for i in range(nr_nodes)}
    feasible_nodes = range(nr_nodes) if lemma != 'PAG' else [i for i in range(nr_nodes) if graphs_without_node_i[i] is not None]
    results = []
    for (X, Y) in it.combinations(feasible_nodes, 2):
        lovo_error_edge, baseline_error_edge, adj_set = lovo_comparison(X, Y, graphs_without_node_i[Y], graphs_without_node_i[X], data.copy(), lemma)
        if not np.isnan(lovo_error_edge) and not m_separated(true_graph, {X}, {Y}, adj_set):
            print(true_graph.edges())
            print(f"Nodes {X} and {Y} are actually not m-separated by {adj_set}")
        else:
            results.append((i, (X, Y), lovo_error_edge, baseline_error_edge))
    return results


if __name__ == '__main__':
    logger = logging.getLogger()
    logger.setLevel(logging.WARNING)

    nr_nodes = 10
    nr_runs = 1000
    nr_potential_edges = math.comb(nr_nodes, 2)

    # Count number of excluded edges
    ps = np.linspace(0.1, 0.9, 9)
    qs = np.linspace(0.1, 0.9, 9)
    settings = {
        'DAG': ('DAG', 0),
        'GX_ADMG_G_DAG': ('GX_ADMG_G_DAG', 0),
        'ADMG': ('ADMG', 0.1),
        'CPDAG': ('CPDAG', 0),
        'PAG': ('PAG', ps, [0]),
        'PAG_varying_q': ('PAG', [0.3], qs),
    }

    for name, params in settings.items():
        graph_type, ps, qs = params
        nr_excluded_edges = []
        for p, q in it.product(ps, qs):
            for i in range(nr_runs):
                if i % 50 == 0:
                    print(i)
                _, coefficient_matrix, _ = simulate_data(nr_nodes, 1, p, q)
                true_graph = ExtendedADMG.from_adjacency_matrix(coefficient_matrix)

                if graph_type == 'CPDAG':
                    graphs_without_node_i = {i: true_graph.project_to_GX(i).to_CPDAG() for i in range(nr_nodes)}
                elif graph_type == 'PAG':
                    graphs_without_node_i = {i: true_graph.project_to_GX(i).to_PAG() for i in range(nr_nodes)}
                else:
                    graphs_without_node_i = {i: true_graph.project_to_GX(i) for i in range(nr_nodes)}
                
                feasible_nodes = range(nr_nodes) if graph_type != 'PAG' else [i for i in range(nr_nodes) if graphs_without_node_i[i] is not None]
                count_excluded_edges = 0
                for (X, Y) in it.combinations(feasible_nodes, 2):
                    graph_without_X = graphs_without_node_i[X]
                    graph_without_Y = graphs_without_node_i[Y]
                    if not edge_might_exist(X, Y, graph_without_Y, graph_without_X, graph_type):
                        if X in true_graph.neighbors(Y):
                            print(true_graph.edges())
                            raise ValueError(f"Edge ({X}, {Y}) exists")
                        count_excluded_edges += 1

                nr_excluded_edges.append(count_excluded_edges)

        df = pd.DataFrame({'p': np.repeat(ps, len(nr_excluded_edges)/len(ps)), 'q': np.repeat(qs, len(nr_excluded_edges)/len(qs)), 'number_excluded_edges': nr_excluded_edges})
        df.to_csv(f'simulation_results/excluded_edges_{name}.csv')
    
    # Parent adjustment based on true marginal graphs
    sample_size = 5000
    p = 0.3 
    settings = {
        'DAG': ('DAG', 0),
        'GX_ADMG_G_DAG': ('GX_ADMG_G_DAG', 0),
        'ADMG': ('ADMG', 0.1),
        'CPDAG': ('CPDAG', 0),
        'PAG': ('PAG', 0)
    }
    for name, (lemma, q) in settings.items():
        print(name)
        all_results = Parallel(n_jobs=-1)(delayed(one_repetition)(i, nr_nodes, sample_size, p, q, lemma) for i in range(nr_runs))
        flat_results = [item for sublist in all_results for item in sublist]
        results_df = pd.DataFrame(flat_results, columns=['sim', 'edge', 'lovo_error', 'baseline_error'])
        results_df.to_csv(f'simulation_results/parent_adj_{name}.csv')
    

