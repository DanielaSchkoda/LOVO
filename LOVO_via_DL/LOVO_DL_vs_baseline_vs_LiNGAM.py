import os.path
import sys

import jax
import numpy as np
import pandas as pd
import lingam
from flax.training import checkpoints
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression
import math
import matplotlib.pyplot as plt
from scipy import signal, stats
import random
from sklearn import preprocessing
from scipy.stats import differential_entropy, pearsonr

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from jax_model.transitive_predictor import TransitivePredictor


def data_complete(sample_size):
    X = np.random.uniform(0,1, sample_size)
    Y = X + np.random.uniform(0,1, sample_size)
    Z = X + Y +  np.random.uniform(0,1, sample_size)
    return np.concatenate((X.reshape(-1, 1),Z.reshape(-1,1), Y.reshape(-1, 1),), axis=1)


def sample_dag():
    arrows = []
    nodes = ['X','Y','Z']
    middle_node = random.sample(nodes,1)[0]
    other_nodes = [node for node in nodes if node != middle_node]
    for node in other_nodes:
        coin = random.sample([0,1],1)
        if coin == 0:
            arrows += [[node, middle_node]]
        else:
            arrows += [[middle_node, node]]
    return arrows


def random_dag_with_two_arrows():
    permutation = list(np.random.permutation(range(3)))
    middle_node = random.sample(range(3),1)[0]
    #nodes = ['X', 'Y', 'Z']
    #arrows = [(0,1), (0,2), (1,2)]
   # arrows.remove(arrows[middle_node])
    #arrows_xyz = []
    #for arrow in arrows:
    #    arrows_xyz.append(nodes[permutation[arrow[0]]] + '->' + nodes[permutation[arrow[1]]])
    return permutation, middle_node # # arrows_xyz


def compute_dag(permutation, middle_node):
    inverse_permutation = np.argsort(permutation)
    nodes = ['X', 'Y', 'Z']
    complete_dag = ['0 -> 1', '0 -> 2', '1 -> 2']
    dag = [edge for edge in complete_dag if str(middle_node) in edge]
    for j in range(3):
        dag = [edge.replace(str(j), nodes[inverse_permutation[j]]) for edge in dag]
    print(dag)
    return dag

def XZ_YZ_relations(permutation, middle_node):
    dag = compute_dag(permutation, middle_node)
    if 'X -> Z' in dag or ('X -> Y' in dag and 'Y -> Z' in dag):
        XZrelation = '->'
    elif 'Z -> X' in dag or ('Z -> Y' in dag and 'Y -> X' in dag):
        XZrelation = '<-'
    elif 'Y -> X' in dag and 'Y -> Z' in dag:
        XZrelation = '<->'
    else:
        XZrelation = 'indep'

    if 'Y -> Z' in dag or ('Y -> X' in dag and 'X -> Z' in dag):
        YZrelation = '->'
    elif 'Z -> Y' in dag or ('Z -> X' in dag and 'X -> Y' in dag):
        YZrelation = '<-'
    elif 'X -> Y' in dag and 'X -> Z' in dag:
        YZrelation = '<->'
    else:
        YZrelation = 'indep'
    return [XZrelation, YZrelation]


def generate_data_from_dag(permutation, middle_node, sample_size):
    XYZ = np.empty((sample_size, 3))
    alpha = np.random.uniform(-1,1,3)
    alpha[middle_node] = 0
    XYZ[:,0] = np.random.uniform(-1,1, sample_size)
    XYZ[:,1] = alpha[2] * XYZ[:,0] + np.random.uniform(-1,1, sample_size)
    XYZ[:,2] = alpha[1] * XYZ[:,0] + alpha[0] * XYZ[:,1] + np.random.uniform(-1,1, sample_size)
    scaler = preprocessing.StandardScaler().fit(XYZ)
    XYZ = scaler.transform(XYZ)
    return XYZ[:,permutation]


def compute_list_of_arrows(permutation, middle_node):
    inverse_permutation =  np.argsort(permutation)
    list_of_nodes = ['X', 'Y', 'Z']
    arrows_with_names = []
    arrows = [[0,1], [0,2], [1,2]]
    arrows = [arrow for arrow in arrows if middle_node in arrow]
    for arrow in arrows:
        arrows_with_names.append(list_of_nodes[inverse_permutation[arrow[0]]] + '->' + list_of_nodes[inverse_permutation[arrow[1]]])
    return arrows_with_names


def generate_data(sample_size, dag):
    if dag == "chainXZY":
        X = np.random.uniform(0,1, sample_size)
        Z = X + np.random.uniform(0,1, sample_size)
        Y = Z +  np.random.uniform(0,1, sample_size)

    if dag == "collider":
        X = np.random.uniform(0,1, sample_size)
        Y = np.random.uniform(0,1, sample_size)
        Z = X + Y + np.random.uniform(0,1, sample_size)

    if dag == "chainXYZ":
        X = np.random.uniform(0,1, sample_size)
        Y = X + np.random.uniform(0,1, sample_size)
        Z = Y + np.random.uniform(0,1, sample_size)

    return np.concatenate((X.reshape(-1, 1),Z.reshape(-1,1), Y.reshape(-1, 1),), axis=1)




def variable_split(XYZ):
    ratios = [0.4, 0.4]
    sample_size = XYZ.shape[0]
    sample_size_XZ =  math.floor(ratios[0] * sample_size)
    sample_size_ZY =  math.floor(ratios[1] * sample_size)
    sample_size_XZY = sample_size - sample_size_XZ - sample_size_ZY
    XZ = XYZ[:sample_size_XZ,[0,2]]
    ZY = XYZ[sample_size_XZ:sample_size_XZ + sample_size_ZY,[1,2]]
    return XZ, ZY, XYZ[sample_size-sample_size_XZY:,:]

def baseline_predictor(XZtrain, ZYtrain, Xtest):
    #clf = MLPRegressor(alpha=1e-05, hidden_layer_sizes=(5, 2), random_state=1,
    #          solver='lbfgs')
    clf = LinearRegression()
    clf.fit(ZYtrain[:,0:1],ZYtrain[:,1])
    Y_predicted = clf.predict(XZtrain[:,1:2])


    #clf = MLPRegressor(alpha=1e-05, hidden_layer_sizes=(5, 2), random_state=1,
    #                   solver='lbfgs')

    clf = LinearRegression()
    clf.fit(XZtrain[:,0:1], Y_predicted)

    print('pred output')
    print(clf.predict([[0],[1]]))

    plt.scatter(XZtrain[:,0],Y_predicted, color='blue')
    plt.title("XZtrain / Y_predicted")
    plt.show()

    return clf.predict(Xtest[:,0:1])








def lingam_predictor(samples_XZ, samples_YZ, x):
    model = lingam.DirectLiNGAM()
    model.fit(samples_XZ)
    adjacency_matrix_XZ = model.adjacency_matrix_
    model.fit(samples_YZ)
    adjacency_matrix_YZ = model.adjacency_matrix_
    return adjacency_matrix_XZ, adjacency_matrix_YZ


def positivity_test(corr_1,corr_2):
    if abs(corr_1) >= abs(corr_2):
        corr_3 = corr_2/corr_1
    else: corr_3 = corr_1/corr_2
    print(corr_3)
    matrix_A = [[1,corr_1,corr_2], [corr_1, 1, 0], [corr_2, 0, 1]]
    matrix_B = [[1,corr_1,corr_2], [corr_1, 1, corr_3], [corr_2, corr_3, 1]]
    print(f" collider possible? {np.all(np.linalg.eigvals(matrix_A) >= 0)}")
    print(f" chain possible? {np.all(np.linalg.eigvals(matrix_B) >= 0)}")


def collider_possible(AB, BC):
    corrAB = stats.pearsonr(AB[:,0], AB[:,1]).statistic
    corrBC = stats.pearsonr(BC[:,0], BC[:,1]).statistic
    covmatrix = [[1,corrAB,corrBC], [corrAB, 1, 0], [corrBC, 0, 1]]
    return np.all(np.linalg.eigvals(covmatrix) >= 0)

def is_not_positive(rhoXZ, rhoYZ):
    covmatrix = [[1,rhoXZ,rhoYZ], [rhoXZ, 1, 0], [rhoYZ, 0, 1]]
    return not np.all(np.linalg.eigvals(covmatrix) >= 0)


def chain_possible(AC, BC):
    clf = LinearRegression()
    clf.fit(AC[:,0:1], AC[:,1])
    noise_AtoC = AC[:,1] - clf.predict(AC[:,0:1])

    clf = LinearRegression()
    clf.fit(BC[:,0:1], BC[:,1])
    noise_BtoC = BC[:,1] - clf.predict(BC[:,0:1])

    bins = np.linspace(-1.4,1.4,100)
    counts_AtoC, _ = np.histogram(noise_AtoC, bins=bins)
    counts_BtoC, _ = np.histogram(noise_BtoC, bins=bins)

    recoveredABC, remainder = signal.deconvolve(counts_AtoC, signal.minimum_phase(counts_BtoC))
    recoveredBAC, remainder = signal.deconvolve(counts_BtoC, signal.minimum_phase(counts_AtoC))
    possible_chains = ['A->B->C', 'B->A->C']
    if np.any(recoveredABC < 0):
        possible_chains.remove('A->B->C')
    if np.any(recoveredBAC < 0):
        possible_chains.remove('B->A->C')
    return possible_chains



def remove_zeroes(list_of_numbers):
    while list_of_numbers[0] == 0:
        list_of_numbers = list_of_numbers[1:]
    while list_of_numbers[-1] ==0:
        list_of_numbers = list_of_numbers[:-1]
    return list_of_numbers


def linear_predictor(x, sigmaX, sigmaY, corrXZ, corrZY, causal_structure):
    if causal_structure == 'chainXZY':
        return x * sigmaY/sigmaX * corrXZ * corrZY
    if causal_structure == 'chainXYZ':
        return x * sigmaY/sigmaX * corrXZ/corrZY
    if causal_structure =='collider':
        return 0

def infer_bivariate_structure(AB):
    model = lingam.RCD()
    model.fit(AB)
    #print(model.adjacency_matrix_)
    if not np.any(model.adjacency_matrix_):
        output_string = 'indep'
    else: output_string = '-'
    if model.adjacency_matrix_[1,0] != 0:
        output_string = output_string +'>'
    if model.adjacency_matrix_[0,1] != 0:
        output_string = '<' + output_string
    return output_string


def transitive_predictor(XZ, YZ, X):
    # requires normalized and zero mean data
    rhoXZ = stats.pearsonr(XZ[:,0],XZ[:,1]).statistic
    rhoYZ = stats.pearsonr(YZ[:,0],YZ[:,1]).statistic
    XZrelation = infer_bivariate_structure(XZ)
    YZrelation = infer_bivariate_structure(YZ)

    print(f"XZrelation = {XZrelation}")
    print(f"YZrelation = {YZrelation}")

    predictor = 'baseline'

    #if XZrelation == '->' and YZrelation == '->':
    #    if abs(rhoXZ) < abs(rhoYZ):
    #        # test for chain X -> Y -> Z
    #        if is_not_positive(rhoXZ, rhoYZ) or (entropy(XZ[:,1] - rhoXZ * XZ[:,0]) > entropy(YZ[:,1] - rhoYZ * YZ[:,0]) and entropy(XZ[:,1] - rhoXZ * XZ[:,0]) < entropy(YZ[:,0])):
    #            predictor = 'Ymediator'
    #        # test for collider X -> Z <- Y
    #        elif entropy(XZ[:,1] - rhoXZ * XZ[:,0]) > entropy(YZ[:,0]) and entropy(XZ[:,1] - rhoXZ * XZ[:,0]) < entropy(YZ[:,1] - rhoYZ * YZ[:,0]):
    #            predictor = 'ignore'
    #    else:
    #        # test for chain Y -> X -> Z
    #        if is_not_positive(rhoXZ, rhoYZ) or (entropy(YZ[:,1] - rhoYZ * YZ[:,0]) > entropy(XZ[:,1] - rhoXZ * XZ[:,0]) and entropy(XZ[:,1] - rhoXZ * XZ[:,0]) < entropy(YZ[:,0])):
    #            predictor = 'Xmediator'
    #        # test for collider X -> Z <- Y
    #        elif entropy(XZ[:,1] - rhoXZ * XZ[:,0]) > entropy(YZ[:,0]) and entropy(YZ[:,1] - rhoYZ * YZ[:,0]) < entropy(XZ[:,1] - rhoXZ * XZ[:,0]):
    #            predictor = 'ignore'


        #if is_not_positive(rhoXZ, rhoYZ):
        #    if abs(rhoXZ) < abs(rhoYZ):
        #        predictor = 'Ymediator'
        #    else:
        #        predictor = 'Xmediator'


        # collider at Z: Z - rhoXZ * X = rhoYZ * Y + ...
        #print('collider test')
        #print(entropy(XZ[:,1] - rhoXZ * XZ[:,0]) > entropy(YZ[:,0]))
        #print(entropy(XZ[:,1] - rhoXZ * XZ[:,0]))
        #print(entropy(YZ[:,0]))

        # chain X -> Y -> Z: Z - rhoXZ * X = Z - rhoYZ * Y + ...
        #print('Ymediator test')
        #print(entropy(XZ[:,1] - rhoXZ * XZ[:,0]) > entropy(YZ[:,1] - rhoYZ * YZ[:,0]))
        #print(entropy(XZ[:,1] - rhoXZ * XZ[:,0]))
        #print(entropy(YZ[:,1] - rhoYZ * YZ[:,0]))

        #plt.hist(XZ[:,1] - rhoXZ * XZ[:,0], bins=30)
        #plt.show()
        #plt.hist(YZ[:,1] - rhoYZ * YZ[:,0], bins=30)
        #plt.show()


        #print(is_convolution(XZ[:,1] - rhoXZ * XZ[:,0], rhoYZ * YZ[:,0]))

        #print(is_convolution(XZ[:,1] - rhoXZ * XZ[:,0], YZ[:,1] - rhoYZ * YZ[:,0]))
        # chain Y -> X -> Z: Z - rhoYZ * XY = Z - rhoXZ * X + ...
        #print(entropy(YZ[:,1] - rhoYZ * YZ[:,0]) > entropy(XZ[:,1] - rhoXZ * XZ[:,0]))
        #print(is_convolution(YZ[:,1] - rhoYZ * YZ[:,0], XZ[:,1] - rhoXZ * XZ[:,0]))

    if XZrelation == '<->' and YZrelation == '->' and abs(rhoXZ) < abs(rhoYZ):
            predictor = 'Ymediator'
    elif XZrelation == '->' and YZrelation == '<->' and  abs(rhoXZ) > abs(rhoYZ):
            predictor = 'Xmediator'


    print(f"predictor = {predictor}")

    if predictor == 'ignore':
        return 0 * X, predictor, XZrelation, YZrelation
    elif predictor == 'baseline':
        return rhoXZ * rhoYZ * X, predictor, XZrelation, YZrelation
    elif predictor == 'Ymediator':
        return rhoXZ/rhoYZ * X, predictor, XZrelation, YZrelation
    elif predictor == 'Xmediator':
        return rhoYZ/rhoXZ * X, predictor, XZrelation, YZrelation



def is_convolution(output, impulse):
    max_values = max(output)
    min_values = min(output)
    bins_output = np.linspace(min_values, max_values, 20)
    bins_impulse = [a + min(output) - min_values for a in bins_output]
    rel_freq_output, _ = np.histogram(output, bins=bins_output)
    rel_freq_impulse, _ = np.histogram(impulse, bins=bins_impulse)
    rel_freq_output = rel_freq_output / len(output)
    rel_freq_impulse = rel_freq_impulse / len(impulse)

    recovered, remainder = signal.deconvolve(rel_freq_output, signal.minimum_phase(rel_freq_impulse))
    plt.bar(range(len(recovered)), recovered, color ='blue')
    plt.show()
    plt.bar(range(len(rel_freq_output)), rel_freq_output, color ='red')
    plt.show()


    #print(f"deconvolution error {error}")
    return not np.any(recovered < 0)


def entropy(samples):
    samples = samples.reshape(-1, 1)
    scaler = preprocessing.StandardScaler().fit(samples)
    samples = scaler.transform(samples)
    return differential_entropy(samples, method='ebrahimi')


def print_scatter_plots(XYZ):
    plt.scatter(XYZ[:,0], XYZ[:,1])
    plt.xlabel('X')
    plt.ylabel("Y")
    plt.show()

    plt.scatter(XYZ[:,0], XYZ[:,2])
    plt.xlabel('X')
    plt.ylabel("Z")
    plt.show()

    plt.scatter(XYZ[:,1], XYZ[:,2])
    plt.xlabel('Y')
    plt.ylabel("Z")
    plt.show()


# Deep Learning Model hyperparameters. TODO store these in checkpoint
NUM_ENC_LAYERS = 3
NUM_DEC_LAYERS = 1
NUM_HEADS = 8  # Number of attention heads.
MODEL_SIZE = 64
DROPOUT_RATE = 0.0
LEARNING_RATE = 1e-4
ACC_GRADS = 1


def correlation_SHD_LOVO():
    error = []
    error_diff = []
    error_base = []
    SHDs = []
    number_of_runs = 1000
    sample_size = 500
    for j in range(number_of_runs):
        print(f"run {j}")
        permutation, middle_node = random_dag_with_two_arrows()
        XYZ = generate_data_from_dag(permutation, middle_node, sample_size)
        XZtrain, YZtrain, XYZtest = variable_split(XYZ)

        prediction, predictor, inferredXZrelation , inferredYZrelation = transitive_predictor(XZtrain, YZtrain, XYZtest[:,0])
        true_relations = XZ_YZ_relations(permutation, middle_node)
        print(f"true XZrelation: {true_relations[0]}, true YZrelation: {true_relations[1]}")
        inferred_relations = [inferredXZrelation, inferredYZrelation]
        SHD = sum( i != j for (i,j) in zip(true_relations, inferred_relations))

        rhoXY = stats.pearsonr(XYZtest[:,0],XYZtest[:,1]).statistic
        rhoXZ = stats.pearsonr(XZtrain[:,0],XZtrain[:,1]).statistic
        rhoYZ = stats.pearsonr(YZtrain[:,0],YZtrain[:,1]).statistic
        rhoXYbase = rhoXZ * rhoYZ
        rhoXYpred = rhoXYbase

        if predictor == 'Xmediator':
            rhoXYpred = rhoYZ / rhoXZ
        elif predictor == 'Ymediator':
            rhoXYpred = rhoXZ / rhoYZ
        error.append(abs(rhoXYpred - rhoXY))
        error_diff.append(abs(rhoXYpred - rhoXY) - abs(rhoXYbase - rhoXY))
        print(f"error: {abs(rhoXYpred - rhoXY)}")
        SHDs.append(SHD)
    return error_diff, SHDs





if __name__ == "__main__":
    error_diff, SHDs = correlation_SHD_LOVO()
    print(pearsonr(error_diff, SHDs))
    # plt.scatter(error_diff, SHDs)
    # plt.show()



    number_of_runs = 1000
    sample_size = 30000

    dl_subsamples = 3000  # This is what the neural net has trained on
    dl_model_state = checkpoints.restore_checkpoint('checkpoint_best', target=None)
    model = TransitivePredictor(
        max_num_nodes=2,
        hidden_dim=MODEL_SIZE,
        depth_encoder=NUM_ENC_LAYERS,
        depth_decoder=NUM_DEC_LAYERS,
        dropout_rate=DROPOUT_RATE, num_heads=NUM_HEADS,
        lr=LEARNING_RATE,
        accumulate_grads=ACC_GRADS
    )

    loss_transitive = []
    loss_baseline = []
    loss_optimal = []
    loss_for_not_baseline_predictors = []
    loss_for_not_baseline_predictors_baseline = []
    error = []
    error_base = []
    error_dl = []
    for j in range(number_of_runs):
        print(f"run {j}")
        permutation, middle_node = random_dag_with_two_arrows()
        #permutation = [0,1,2]
        #middle_node = 2
        #print(permutation)
        #print(middle_node)
        print(compute_list_of_arrows(permutation, middle_node))
        XYZ = generate_data_from_dag(permutation, middle_node, sample_size)

        #model = lingam.RCD()
        #model.fit(XYZ)
        #print(model.adjacency_matrix_)
        XZtrain, YZtrain, XYZtest = variable_split(XYZ)

        prediction, predictor,_ ,_ = transitive_predictor(XZtrain, YZtrain, XYZtest[:,0])
        rhoXY = stats.pearsonr(XYZtest[:,0],XYZtest[:,1]).statistic
        rhoXZ = stats.pearsonr(XZtrain[:,0],XZtrain[:,1]).statistic
        rhoYZ = stats.pearsonr(YZtrain[:,0],YZtrain[:,1]).statistic
        rhoXYbase = rhoXZ * rhoYZ
        rhoXYpred = rhoXYbase
        sub_idx = np.random.choice(range(XZtrain.shape[0]), dl_subsamples, replace=False)
        XZsub = XZtrain[sub_idx, :]
        sub_idx = np.random.choice(range(YZtrain.shape[0]), dl_subsamples, replace=False)
        YZsub = YZtrain[sub_idx, :]
        rhoXYdl = model.apply({'params': dl_model_state['params']}, XZsub, YZsub).item()

        if predictor == 'Xmediator':
            rhoXYpred = rhoYZ / rhoXZ
        elif predictor == 'Ymediator':
            rhoXYpred = rhoXZ / rhoYZ

        #relative_error.append(abs(rhoXYpred - rhoXY) / abs(rhoXY))
        #relative_error_base.append(abs(rhoXYbase - rhoXY) / abs(rhoXY))
        error.append(abs(rhoXYpred - rhoXY))
        error_base.append(abs(rhoXYbase - rhoXY))


        #plt.scatter(XYZtest[:,0], XYZtest[:,1], color = 'blue')
        #plt.scatter(XYZtest[:,0], prediction, color ='red')
        #plt.scatter(XYZtest[:,0], rhoXY * XYZtest[:,0], color ='green')
        #plt.show()

        loss_tr = sum((prediction - XYZtest[:,1])**2)/sample_size

        rhoYZ = stats.pearsonr(YZtrain[:,0],YZtrain[:,1]).statistic

        loss_base = sum((XYZtest[:,0] * rhoXZ * rhoYZ - XYZtest[:,1])**2)/sample_size
        loss_opt =  sum((XYZtest[:,0] * rhoXY - XYZtest[:,1])**2)/sample_size


        loss_transitive.append(loss_tr)
        loss_baseline.append(loss_base)
        loss_optimal.append(loss_opt)
        print(f"loss_tr: {loss_tr}")
        print(f"loss_base: {loss_base}")
        print(f"loss_opt: {loss_opt}")

        if predictor != 'baseline':
            loss_for_not_baseline_predictors.append(loss_tr)
            loss_for_not_baseline_predictors_baseline.append(loss_base)

    plt.scatter(loss_baseline,loss_transitive, color='brown')
    plt.show()
    #plt.scatter(loss_optimal, loss_baseline, color='red')
    #plt.scatter(loss_optimal, loss_optimal, color='yellow')
    #plt.scatter(loss_baseline,loss_transitive, color='brown')
    #plt.scatter(loss_baseline, loss_baseline, color='red')
    df = pd.DataFrame({'error_base': error_base, 'error_rcd': error, 'error_dl': error_dl})
    df.to_csv('pred_error_res.csv')
    plt.scatter(error_base, error_base, color = 'black', label='Baseline')
    plt.scatter(error_base, error, color='blue', label='RCD')
    plt.scatter(error_base, error_dl, color = 'orange', label='Deep Learning')
    plt.xlabel('baseline loss')
    plt.ylabel('LOVO loss')
    plt.legend()
    plt.savefig('pred_error.png')
    plt.show()
    print(f"baseline loss = {sum(loss_baseline)/sample_size}")
    print(f"transitive loss = {sum(loss_transitive)/sample_size}")
    print(f"optimal loss = {sum(loss_optimal)/sample_size}")

    print('error base: ', np.mean(error_base))
    print('error rcd: ', np.mean(error))
    print('error dl: ', np.mean(error_dl))
    print(sum(x.size for x in jax.tree_leaves(dl_model_state['params'])))

    #print(f" non baseline / baseline = {sum(loss_for_not_baseline_predictors)/sum(loss_for_not_baseline_predictors_baseline)}")
