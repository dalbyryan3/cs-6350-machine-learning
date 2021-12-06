# %%
# Libraries 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Personal Libraries
from neural_network import NeuralNetwork

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
# Checking if network can overfit data
network = NeuralNetwork(100)
total_MSE_loss_by_epoch = network.train(X_train, y_train, r=0.0001, T=300) 
plt.figure()
plt.plot(total_MSE_loss_by_epoch, label='Training MSE loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss vs Epoch')
plt.show()

y_train_pred = network.predict(X_train)
train_err = calculate_prediction_error(y_train_pred, y_train)
y_test_pred = network.predict(X_test)
test_err = calculate_prediction_error(y_test_pred, y_test)

print('Training error = {0} and testing error = {1}'.format(train_err, test_err))

# %%
r0 = 0.001
d = 1
T = 100
t_vals = np.arange(T)+1
r_sched = r0 / (1 + (r0/d)*t_vals)

H_vals = np.array([5, 10, 25, 50, 100])

for H in H_vals:
    network = NeuralNetwork(H)
    total_MSE_loss_by_epoch = network.train(X_train, y_train, random_weight_initialization=True, r_sched=r_sched)
    plt.figure()
    plt.plot(total_MSE_loss_by_epoch, label='Training MSE loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss vs Epoch')
    plt.show()

    y_train_pred = network.predict(X_train)
    train_err = calculate_prediction_error(y_train_pred, y_train)
    y_test_pred = network.predict(X_test)
    test_err = calculate_prediction_error(y_test_pred, y_test)
    print('H = {0}, gamma_0 = {1}, d = {2}: training error = {3} and testing error = {4}'.format(H, r0, d, train_err, test_err))


# Results- can train very well, but training highly susceptible to hyperparameters
# %%
# TODO: 
# Check gradients
# Tune training
# Document and analyze
