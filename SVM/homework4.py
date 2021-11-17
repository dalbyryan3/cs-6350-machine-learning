# %%
# Libraries 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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

# Displaying train and test errors and plotting function for 2
def visualize_and_get_results_prob2(models, obj_func_vals_by_epoch,  C_vals):
    plt.figure()
    for i, C in enumerate(C_vals):
        print("C = {0}".format(C))
        model = models[i]
        print("Weight vector = {0}\n bias = {1}".format(model.w[:-1], model.w[-1]))
        train_err = calculate_prediction_error(model.primal_predict(X_train), y_train)
        test_err = calculate_prediction_error(model.primal_predict(X_test), y_test)
        print("Training error = {0}, Test error = {1}".format(train_err, test_err))
        print()
        plt.plot(obj_func_vals_by_epoch[i][1:], label='Training obj func value, C={0}'.format(C)) # Note that not plotting first epoch value so results are easier to visualize on plot
    plt.title('SVM Primal Objective Value vs Epoch for Stochastic Gradient Descent')
    plt.xlabel('Number of Stochastic Gradient Epochs')
    plt.ylabel('SVM Primal Objective Value')
    plt.legend()
    plt.show()

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
# 2a plotting and visualization
visualize_and_get_results_prob2(svm_primal_sgd_models_2a, svm_primal_sgd_obj_func_vals_by_epoch_2a, C_vals)

# %%
# 2b
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
# 2b plotting and visualization
visualize_and_get_results_prob2(svm_primal_sgd_models_2b, svm_primal_sgd_obj_func_vals_by_epoch_2b, C_vals)
