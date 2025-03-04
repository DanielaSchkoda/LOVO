import numpy as np
import itertools as it
from scipy.linalg import inv
from extended_pywhy_graphs import ExtendedADMG
from LOVO_via_parent_adjustment import three_step_LOVO_predictor
import math
from simulate_data import simulate_data
import random
import pandas as pd

def project(A, leave_out, top_order, children_leave_out, swap_columns = False):
    '''
    Project the adjacency matrix A to the matrix A_X corresponding to leaving out a specific node. Note that in the returned matrix,
    we do not leave out the row and potentially the column corresponding to the node to leave out. Instead, we set them to nan, to
    to ensure that the indices in A_X still correspond to the indices in the original adjacency matrix A.

    Parameters:
    A (numpy.ndarray): The original adjacency matrix.
    leave_out (int): The index of the node to leave out.
    top_order (list): The topological order of all nodes.
    children_leave_out (list): The children nodes of the node to leave out.
    swap_columns (bool, optional): Whether to swap columns of the node to leave out and its oldest child. Defaults to False.

    Returns:
    numpy.ndarray: The adjusted adjacency matrix after leaving out the specified node.
    '''
    A_X = A.copy()
    A_X[leave_out] = np.full(A_X.shape[1], np.nan)
    if len(children_leave_out) < 2:
        A_X[:, leave_out] = np.full(A_X.shape[0], np.nan)
        return A_X
    
    # Scale the latent column. The edge to its oldest child should be 1.
    oldest_child = next(i for i in top_order if i in children_leave_out)
    A_X[:, leave_out] /= A_X[oldest_child, leave_out]

    # To create an adverserial test case, we swap the latent variables.
    if swap_columns and oldest_child_is_unique(oldest_child, children_leave_out, A):
        A_X[:, [leave_out, oldest_child]] = A_X[:, [oldest_child, leave_out]]
    
    return A_X

def oldest_child_is_unique(oldest_child, children_X, A_X):
    return all(not np.isclose(A_X[i, oldest_child], 0) for i in children_X)

def reconstruct_A(X, Y, A_X, A_Y, G_X, G_Y, top_order_XZ, top_order_YZ):
    '''
    Reconstruct the adjacency matrix A from the projected matrices A_X and A_Y according to Theorem 5. In theory, this would be possible solely based on the
    projected matrices A_X and A_Y. However, to avoid to repeatedly infer G_X, G_Y, and the corresponding topological orders, from A_X, A_Y, we pass them as arguments. 
    '''
    p = A_X.shape[0]
    Z = [W for W in range(p) if W != X and W != Y]
    # Bring overlapping part to same scale
    norms_X = np.linalg.norm(A_X[Z], axis=0)
    norms_Y = np.linalg.norm(A_Y[Z], axis=0)
    norms_X[np.isclose(norms_X, 0) | np.isnan(norms_X)] = 1
    norms_Y[np.isclose(norms_Y, 0) | np.isnan(norms_Y)] = 1
    A_X /= norms_X
    A_Y /= norms_Y
    
    # Find second options for A_X and A_Y if they exist (compare idenfifiability case a) in proof of Theorem 5)
    options_A_X = [A_X]
    if not np.isnan(A_X[:, Y]).all():
        children_Y = [c for e in G_X.bidirected_edges for c in e]
        oldest_child_latent = next(i for i in top_order_XZ if i in children_Y) 
        if oldest_child_is_unique(oldest_child_latent, children_Y, A_Y):
            second_option = A_X.copy()
            second_option[:,[Y, oldest_child_latent]] = second_option[:,[oldest_child_latent, Y]]
            options_A_X.append(second_option)

    options_A_Y = [A_Y]
    if not np.isnan(A_Y[:, X]).all():
        children_X = [c for e in G_Y.bidirected_edges for c in e]
        oldest_child_latent = next(i for i in top_order_YZ if i in children_X)
        if oldest_child_is_unique(oldest_child_latent, children_X, A_X):
            second_option = A_Y.copy()
            second_option[:,[X, oldest_child_latent]] = second_option[:,[oldest_child_latent, X]]
            options_A_Y.append(second_option)
    
    all_options = list(it.product(options_A_X, options_A_Y))
    distances = np.zeros(len(list(all_options)))
    for i, (A_X, A_Y) in enumerate(all_options):
        # To same sign
        scalar_products = np.sign(np.einsum('ij,ij->j', A_X[Z], A_Y[Z]))
        scalar_products[np.isclose(scalar_products, 0) | np.isnan(scalar_products)] = 1
        A_Y *= scalar_products
        distances[i] = np.linalg.norm(np.nan_to_num(A_X[Z] - A_Y[Z], nan=0)) 
        # Update list to contain matrices with aligned sign
        all_options[i] = (A_X.copy(), A_Y.copy())

    sorted_distances, corresponding_indices = np.sort(distances), np.argsort(distances)
    best_distance, best_index = sorted_distances[0], corresponding_indices[0]
    
    if best_distance > 0.5:
        print('Warning: best distance is large: ', best_distance)

    if len(sorted_distances) > 1 and np.isclose(sorted_distances[0], sorted_distances[1]):
        return np.full((p, p), np.nan)
    best_AX, best_AY = all_options[best_index]
    A_hat = combine_to_A(X, Y, best_AX, best_AY, G_X, G_Y)
    return A_hat

def combine_to_A(X, Y, A_X, A_Y, G_X, G_Y):
    '''
    Combines the two already aligned marginal path matrices A_X and A_Y into a single matrix A_hat.

    Parameters:
    X (int): First leave-out variable.
    Y (int): Second leave-out variable.
    A_X (np.ndarray): Marginal path matrix with X.
    A_Y (np.ndarray): Marginal path matrix with Y.
    G_X (np.ndarray): Marginal graph with X.
    G_Y (np.ndarray): Marginal graph with Y.

    Returns:
    np.ndarray: The combined path matrix A_hat.
    '''
    A_hat = np.nanmean([A_X, A_Y], axis=0) 
    if any(np.isclose(np.diag(A_hat), 0)):
        print('Warning: A_hat has a zero on the diagonal, this should not happen for valid A_X, A_Y.')
        raise ValueError('A_hat has a zero on the diagonal')
    A_hat /= np.nan_to_num(np.diag(A_hat), nan=1)

    # In some cases we can still infer the missing value.
    if np.isnan(A_hat[X, Y]) or np.isnan(A_hat[Y,X]):
        A_hat = fill_nans(G_X, G_Y, X, Y, A_hat)
    return A_hat

def fill_nans(G_X, G_Y, X, Y, A_hat):
    '''
    Infer A_hat[X, Y] if Y has at most one child, and A_hat[Y, X] if X has at most one child.
    '''
    ch_X_GX = list(G_X.children(X))
    ch_Y_GY = list(G_Y.children(Y))
    X_multiple_children = len(G_Y.bidirected_edges) > 0 # X_multiple_children iff A_hat[Y, X] already filled
    Y_multiple_children = len(G_X.bidirected_edges) > 0 # Y_multiple_children iff A_hat[X, Y] already filled
    confounded_nodes_G_X = [u for e in G_X.bidirected_edges for u in e]
    confounded_nodes_G_Y = [u for e in G_Y.bidirected_edges for u in e]
    if Y in confounded_nodes_G_Y:
        # 'X -> Y', A_hat[X, Y] not filled so far
        A_hat[X, Y] = 0 
    if X in confounded_nodes_G_X:
        # 'Y -> X', A_hat[Y, X] not filled so far
        A_hat[Y, X] = 0
    if X_multiple_children and not Y_multiple_children and len(ch_Y_GY) > 1:
        # 'Y -> X -> C', A_hat[X, Y] not filled so far
        C = ch_Y_GY[0]
        A_hat[X, Y] = A_hat[C, Y] / A_hat[C, X]
    if Y_multiple_children and not X_multiple_children and len(ch_X_GX) > 1:
        # 'X -> Y -> C', A_hat[Y, X] not filled so far
        C = ch_X_GX[0]
        A_hat[Y, X] = A_hat[C, X] / A_hat[C, Y]
    if not X_multiple_children and not Y_multiple_children:            
        if ch_X_GX == ch_Y_GY:
            pa_X = set(G_X.parents(X))
            pa_Y = set(G_Y.parents(Y))
            # Both have same child in the marginal graphs, Lemma 3 (3c-d)
            if len(ch_X_GX) == 1:
                C = ch_X_GX[0]
                pa_C_GX = set(G_X.parents(C))
                pa_C_GY = set(G_Y.parents(C))
                if pa_X == pa_Y:
                    # X -> Y or Y -> X or X -> C <- Y possible, can't determine the causal effects.
                    return A_hat
                elif pa_X.issubset(pa_Y & pa_C_GY) and pa_Y.issubset(pa_C_GX):
                    # X -> Y -> C or X -> C <- Y possible, can't determine A_hat[Y, X].
                    A_hat[X, Y] = 0
                    return A_hat
                elif pa_Y.issubset(pa_X & pa_C_GX) and pa_X.issubset(pa_C_GY):
                    # Y -> X -> C or Y -> C <- X possible, can't determine A_hat[X, Y].
                    A_hat[Y, X] = 0
                    return A_hat
                elif pa_X.issubset(pa_Y) and pa_Y.issubset(pa_X | pa_C_GX):
                    # X -> Y -> C
                    A_hat[Y, X] = A_hat[C, X] / A_hat[C, Y]
                    A_hat[X, Y] = 0
                    return A_hat
                elif pa_Y.issubset(pa_X) and pa_X.issubset(pa_Y | pa_C_GY):
                    # Y -> X -> C
                    A_hat[X, Y] = A_hat[C, Y] / A_hat[C, X]
                    A_hat[Y, X] = 0
                    return A_hat
            else:
                # Both sinks in the marginal graphs
                if pa_X == pa_Y:
                    # X -> Y and Y -> X possible, can't determine the causal effects.
                    return A_hat
                elif pa_X.issubset(pa_Y):
                    # X -> Y or no edge. Since Y has no child, A_hat[X, Y] = 0. A_hat[Y, X] can't be determined
                    A_hat[X, Y] = 0
                    return A_hat
                elif pa_Y.issubset(pa_X):
                    # Y -> X or no edge. Since X has no child, A_hat[Y, X] = 0. A_hat[X, Y] can't be determined
                    A_hat[Y, X] = 0
                    return A_hat
        # In all other cases, Lemma 3 yields that X and Y are not connected by an edge. They might be connected by one or multiple paths.
        # We use that each of these paths would need to pass through the single child C, D of X, Y, respectively. 
        if len(ch_X_GX) == 0:
            A_hat[Y, X] = 0
        elif len(ch_X_GX) == 1:
            # X -> C maybe -> ... -> Y
            C = list(ch_X_GX)[0]
            A_hat[Y, X] = A_hat[C, X] * A_hat[Y, C]
        if len(ch_Y_GY) == 0:
            A_hat[X, Y] = 0
        elif len(ch_Y_GY) == 1:
            # Y -> D maybe -> ... -> X
            D = list(ch_Y_GY)[0]
            A_hat[X, Y] = A_hat[D, Y] * A_hat[X, D]
    return A_hat

def adjacency_to_path_matrix(adjacency):
    p = adjacency.shape[0]
    return np.linalg.inv(np.eye(p)-adjacency)

def path_matrix_to_adjacency(path_matrix, tol=1e-10):
    try:
        p = path_matrix.shape[0]
        adj = -inv(path_matrix) + np.eye(p)
        adj[np.isclose(adj, 0, atol=tol)] = 0
    except Exception:
        adj = np.full(path_matrix.shape, np.nan)
    return adj

def calculate_corr_from_adjacency(adjacency, X, Y, A, data_XZ, data_YZ):
    '''
    Calculate the correlation between X and Y given the adjacency matrix A. If X is an ancestor of Y, we adjust using the parents of Y,
    and vice versa.
    '''
    p = adjacency.shape[0]
    XZ, YZ = [i for i in range(p) if i != Y], [i for i in range(p) if i != X]
    var_X, var_Y = data_XZ[X].var(), data_YZ[Y].var()
    if not np.isclose(A[Y,X], 0): # X is an ancestor of Y
        cov_XZ = data_XZ.cov()[X].to_numpy()
        cov_XY = adjacency[Y, XZ] @ cov_XZ
    elif not np.isclose(A[X,Y], 0):
        cov_YZ = data_YZ.cov()[Y].to_numpy()
        cov_XY = adjacency[X, YZ] @ cov_YZ
    else:
        # If X is neither an ancestor of Y nor vice versa, take the average of the above approaches.
        cov_XZ = data_XZ.cov()[X].to_numpy()
        cov_YZ = data_YZ.cov()[Y].to_numpy()
        cov_XY = np.mean([adjacency[Y, XZ] @ cov_XZ, adjacency[X, YZ] @ cov_YZ], axis = 0)
    return cov_XY / np.sqrt(var_Y * var_X)


if __name__ == '__main__':
    import warnings
    # Surpress warnings, which occur when taking the mean of an all-nan slice.
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    p = 10
    n = 5000
    nr_runs = 1000
    errors_LOVO = []
    errors_baseline = []
    sims = []

    for i in range(nr_runs):
        print(i)
        data, adjacency, top_order = simulate_data(p, n, dir_edge_density=0.3, bidir_edge_density=0)
        G = ExtendedADMG.from_adjacency_matrix(adjacency)
        graphs_without_node_i = {i: G.project_to_GX(i) for i in range(p)}
        A = adjacency_to_path_matrix(adjacency)
        projected_matrices = {i: project(A, i, top_order, set(G.children(i)), swap_columns = random.choice([True, False])) for i in range(p)}
        for (X, Y) in it.combinations(range(p),2):
            G_X, G_Y = graphs_without_node_i[Y], graphs_without_node_i[X]
            A_X, A_Y = projected_matrices[Y], projected_matrices[X]
            top_order_XZ, top_order_YZ = [i for i in top_order if i != Y], [i for i in top_order if i != X]
            A_hat = reconstruct_A(X, Y, A_X, A_Y, G_X, G_Y, top_order_XZ, top_order_YZ)
            if not np.isnan(A_hat).any():
                adj_hat = path_matrix_to_adjacency(A_hat)
                size_train = math.floor(n/3)
                data_XZ = data.drop([Y], axis=1).iloc[:size_train,:].copy()
                data_YZ = data.drop([X], axis=1).iloc[size_train:2*size_train,:].copy()
                data_XY = data[[X, Y]].iloc[2*size_train:,:].to_numpy()

                LiNGAM_pred = calculate_corr_from_adjacency(adj_hat, X, Y, A_hat, data_XZ, data_YZ)
                true_coeff = np.corrcoef(data_XY, rowvar=False)[0,1]
                errors_LOVO.append(np.abs(LiNGAM_pred-true_coeff))  
                # Scale data to have unit variance such that the baseline indeed estimates the correlation.
                data_XZ[X] = data_XZ[X]/np.std(data_XZ[X])
                data_YZ[Y] = data_YZ[Y]/np.std(data_YZ[Y])
                Z = [i for i in range(p) if i != X and i != Y]
                errors_baseline.append(np.abs(three_step_LOVO_predictor(X, Y, data_XZ, data_YZ, Z) - true_coeff))
                sims.append(i)

    results = pd.DataFrame({'lovo_error': errors_LOVO, 'baseline_error': errors_baseline, 'sim': sims})
    results.to_csv('simulation_results/LOVO_via_LiNGAM.csv')
    
            