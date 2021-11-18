# %%
# Libraries 
from typing import overload
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle

# Personal Libraries
from SVM import Svm

# %%
# General Functions
def convert_0_to_minus1(y):
    """ Convert 0s to -1 in an ndarray.

    Args:
        y (ndarray): Labels with 1s and 0s.

    Returns:
        ndarray: Labels with -1s as 0s.
    """
    ycp = y.copy()
    ycp[ycp==0] = -1
    return ycp

def calculate_prediction_error(y_pred, y):
    """ Get prediction error.

    Args:
        y_pred (ndarray): Predicted label data. A m shape 1D numpy array where m is the number of examples.
        y (ndarray): Ground truth label data. A m shape 1D numpy array where m is the number of examples.

    Returns:
        float: Prediction error.
    """
    return np.sum(y_pred!=y) / y.shape[0]

# Displaying train errors, test errors, and plotting for 2
def output_results_prob2(models, obj_func_vals_by_epoch, C_vals):
    plt.figure()
    for i, C in enumerate(C_vals):
        print("C = {0}".format(C))
        model = models[i]
        print("Weight vector = {0}\n bias = {1}".format(model.w_aug[:-1], model.w_aug[-1]))
        train_err = calculate_prediction_error(model.predict(X_train), y_train)
        test_err = calculate_prediction_error(model.predict(X_test), y_test)
        print("Training error = {0}, Test error = {1}".format(train_err, test_err))
        print()
        plt.plot(obj_func_vals_by_epoch[i][1:], label='Training obj func value, C={0}'.format(C)) # Note that not plotting first epoch value so results are easier to visualize on plot
    plt.title('SVM Primal Objective Value vs Epoch for Stochastic Gradient Descent')
    plt.xlabel('Number of Stochastic Gradient Epochs')
    plt.ylabel('SVM Primal Objective Value')
    plt.legend()
    plt.show()

# Displaying parameters, train errors, and test errors for 3a
def output_results_prob3a(models, C_vals):
    for i, C in enumerate(C_vals):
        print("C = {0}".format(C))
        model = models[i]
        print("Weight vector = {0}\n bias = {1}".format(model.w_aug[:-1], model.w_aug[-1]))
        train_err = calculate_prediction_error(model.predict(X_train), y_train)
        test_err = calculate_prediction_error(model.predict(X_test), y_test)
        print("Training error = {0}, Test error = {1}".format(train_err, test_err))
        print()

# Displaying parameters, train errors, and test errors for 3b
def output_results_prob3b(models, param_vals):
    for i, tup in enumerate(param_vals):
        C = tup[0]
        rbf_gamma = tup[1]
        print("C = {0}, rbf_gamma = {1}".format(C, rbf_gamma))
        model = models[i]
        train_err = calculate_prediction_error(model.predict(X_train), y_train)
        test_err = calculate_prediction_error(model.predict(X_test), y_test)
        print("Training error = {0}, Test error = {1}".format(train_err, test_err))
        print()

def calculate_num_support_vectors_3c(alpha_stars):
    num_support_vectors_list = []
    for alpha_star in alpha_stars:
        num_support_vectors_list.append(np.sum(alpha_star != 0.0))
    return np.array(num_support_vectors_list)

def display_num_support_vectors_3c(num_support_vectors, param_vals):
    for i, tup in enumerate(param_vals):
        C = tup[0]
        rbf_gamma = tup[1]
        print("C = {0}, rbf_gamma = {1}".format(C, rbf_gamma))
        print("Number of support vectors = {0}".format(num_support_vectors[i]))
        print()

def overlapping_consecutive_support_vectors_3c(alpha_stars, param_vals, C_to_analyze=500/873):
    # Assumes that rbf_gamma values in param_vals are ordered in way that overlapping should be analyzed
    overlapping_list = []
    last_rbf_gamma = None
    last_alpha_star = None
    for i, tup in enumerate(param_vals):
        C = tup[0]
        rbf_gamma = tup[1]
        if C == C_to_analyze:
            if last_rbf_gamma is None or last_alpha_star is None:
                last_alpha_star = alpha_stars[i]
                last_rbf_gamma = rbf_gamma
                continue
            overlapping_list.append(((last_rbf_gamma, rbf_gamma), np.sum(alpha_stars[i]==last_alpha_star)))
            last_alpha_star = alpha_stars[i]
            last_rbf_gamma = rbf_gamma
    return overlapping_list

def display_overlapping_consecutive_support_vectors_3c(overlapping, C_to_analyze=500/873):
    print("For C = {0}".format(C_to_analyze))
    for tup in overlapping:
        last_gamma = tup[0][0]
        gamma = tup[0][1]
        num_overlap = tup[1]
        print("Last gamma = {0}, gamma = {1}, num_overlap = {2}".format(last_gamma, gamma, num_overlap))

# %%
data_root_path = './bank-note'
train_data = pd.read_csv('{0}/train.csv'.format(data_root_path), header=None)
test_data = pd.read_csv('{0}/test.csv'.format(data_root_path), header=None)
X_train = train_data.iloc[:, 0:4].to_numpy()
y_train = convert_0_to_minus1(train_data.iloc[:, 4].to_numpy())
X_test = test_data.iloc[:, 0:4].to_numpy()
y_test = convert_0_to_minus1(test_data.iloc[:, 4].to_numpy())
# %%
# 2a
T = 100
r_0 = 0.1
a = 0.1
t_vals = np.arange(T)+1

# Scheduled learning rates
r_sched = r_0 / (1 + t_vals*r_0/a) 

# C values to explore
C_vals_denom = len(y_train)+1
C_vals = [100/C_vals_denom, 500/C_vals_denom, 700/C_vals_denom]
svm_primal_sgd_models_2a = []
svm_primal_sgd_obj_func_vals_by_epoch_2a = []

for C in C_vals:
    svm_primal_sgd = Svm()
    objective_func_vals_by_epoch = svm_primal_sgd.train_primal_stochastic_subgradient_descent(X_train, y_train, T=T, C=C, r_sched=r_sched)
    svm_primal_sgd_models_2a.append(svm_primal_sgd)
    svm_primal_sgd_obj_func_vals_by_epoch_2a.append(objective_func_vals_by_epoch)

# %%
# 2a 
# Plotting and output results
output_results_prob2(svm_primal_sgd_models_2a, svm_primal_sgd_obj_func_vals_by_epoch_2a, C_vals)

# %%
# 2b- LONG RUNNING
T = 100
r_0 = 0.1
a = 0.1
t_vals = np.arange(T)+1

# Scheduled learning rates
r_sched = r_0 / (1 + t_vals)

# C values to explore
C_vals_denom = len(y_train)+1
C_vals = [100/C_vals_denom, 500/C_vals_denom, 700/C_vals_denom]
svm_primal_sgd_models_2b = []
svm_primal_sgd_obj_func_vals_by_epoch_2b = []

for C in C_vals:
    svm_primal_sgd = Svm()
    objective_func_vals_by_epoch = svm_primal_sgd.train_primal_stochastic_subgradient_descent(X_train, y_train, T=T, C=C, r_sched=r_sched)
    svm_primal_sgd_models_2b.append(svm_primal_sgd)
    svm_primal_sgd_obj_func_vals_by_epoch_2b.append(objective_func_vals_by_epoch)

# %%
# 2b 
# Plotting and output results
output_results_prob2(svm_primal_sgd_models_2b, svm_primal_sgd_obj_func_vals_by_epoch_2b, C_vals)

# %%
# 3a
# C values to explore
C_vals_denom = len(y_train)+1
C_vals = [100/C_vals_denom, 500/C_vals_denom, 700/C_vals_denom]
svm_dual_models_3a = []
svm_dual_alpha_stars_3a = []

svm_dual = Svm()
for C in C_vals:
    svm_dual = Svm()
    alpha_star = svm_dual.train_dual(X_train, y_train, C=C)
    svm_dual_models_3a.append(svm_dual)
    svm_dual_alpha_stars_3a.append(alpha_star)

# %%
# 3a 
# Save results 
with open('hw4_3a.pkl', 'wb') as f:
    pickle.dump([svm_dual_models_3a, svm_dual_alpha_stars_3a], f)

# %%
# 3a 
# Loading results 
with open('hw4_3a.pkl', 'rb') as f:
    svm_dual_models_3a, svm_dual_alpha_stars_3a = pickle.load(f)

# %%
# 3a 
# Output results 
output_results_prob3a(svm_dual_models_3a, C_vals)

# %%
# 3b- LONG RUNNING
# C values to explore
C_vals_denom = len(y_train)+1
# C_vals = [100/C_vals_denom]
C_vals = [100/C_vals_denom, 500/C_vals_denom, 700/C_vals_denom]
# rbf_gamma_vals = [0.1]
rbf_gamma_vals = [0.1, 0.5, 1, 5, 100]
svm_dual_models_3b = []
svm_dual_alpha_stars_3b = []
svm_dual_param_vals_3b = []

svm_dual = Svm()
for C in C_vals:
    for rbf_gamma in rbf_gamma_vals:
        svm_dual = Svm()
        alpha_star = svm_dual.train_dual(X_train, y_train, C=C, rbf_kernel=True, rbf_gamma=rbf_gamma)
        svm_dual_models_3b.append(svm_dual)
        svm_dual_alpha_stars_3b.append(alpha_star)
        svm_dual_param_vals_3b.append((C, rbf_gamma))

# %%
# 3b 
# Save results 
with open('hw4_3b.pkl', 'wb') as f:
    pickle.dump([svm_dual_models_3b, svm_dual_alpha_stars_3b, svm_dual_param_vals_3b], f)

# %%
# 3b 
# Loading results 
with open('hw4_3b.pkl', 'rb') as f:
    svm_dual_models_3b, svm_dual_alpha_stars_3b, svm_dual_param_vals_3b = pickle.load(f)

# %%
# 3b 
# Output results 
output_results_prob3b(svm_dual_models_3b, svm_dual_param_vals_3b)

# %%
# 3c
num_support_vectors_vec_3b = calculate_num_support_vectors_3c(svm_dual_alpha_stars_3b)
display_num_support_vectors_3c(num_support_vectors_vec_3b, svm_dual_param_vals_3b)
overlapping_3b = overlapping_consecutive_support_vectors_3c(svm_dual_alpha_stars_3b, svm_dual_param_vals_3b)
display_overlapping_consecutive_support_vectors_3c(overlapping_3b)