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
# Testing
# Check values against neural network in paper problems
score_2a_paper_problems = np.array([[-2.437]])
cache_2a_paper_problems = (np.array([[-6, 6]]), np.array([[0.00247, 0.9975]]), np.array([[-4, 4]]), np.array([[0.01803, 0.9820]]))

backward_vals_paper_problems = np.array([[0.00105, 0.00158], [0.00105, 0.00158]]), np.array([0.00105, 0.00158]), np.array([[-0.0003017, 0.000226], [-0.1217, 0.0910]]), np.array([-0.122, 0.09125]), np.array([[-0.06197], [-3.375]]), np.array([-3.4369])

network_2a = NeuralNetwork(2)
X_2a = np.array([[1,1]]) # bias is already accounted for in implementation
y_2a = np.array([1])
m_2a, d_2a = X_2a.shape
network_2a.W1 = np.array([[-2,2],[-3,3]]) # (dxH)
network_2a.b1 = np.array([-1, 1]) # (H)
network_2a.W2 = np.array([[-2,2],[-3,3]]) # (HxH)
network_2a.b2 = np.array([-1, 1]) # (H)
network_2a.W3 = np.array([[2],[-1.5]]) # (Hx1)
network_2a.b3 = np.array([-1]) # (1)

score_2a, cache_2a = network_2a._forward_pass(X_2a)
backward_vals = network_2a._backwards_pass(X_2a, y_2a, score_2a, cache_2a)

print('Forward pass paper problems:\nscore = {0},\nS1 = {1}, \nZ1 = {2},\nS2 = {3},\nZ2 = {4}\n\n'.format(score_2a_paper_problems, *cache_2a_paper_problems))
print('Forward pass results:\nscore = {0},\nS1 = {1}, \nZ1 = {2},\nS2 = {3},\nZ2 = {4}\n\n'.format(score_2a, *cache_2a))

print('Backward pass paper problems:\ndW1 = {0},\ndb1 = {1}, \ndW2 = {2},\ndb2 = {3},\ndW3 = {4},\ndb3 = {5}\n\n'.format(*backward_vals_paper_problems))
print('Backward pass results:\ndW1 = {0},\ndb1 = {1}, \ndW2 = {2},\ndb2 = {3},\ndW3 = {4},\ndb3 = {5}\n\n'.format(*backward_vals))

# %%
# 2b
# Testing
# Checking if network can overfit data
network_overfit = NeuralNetwork(100)
MSE_by_epoch = network_overfit.train(X_train, y_train, r=0.0005, T=300) 
plt.figure()
plt.plot(MSE_by_epoch, label='Training Set MSE Loss')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.title('MSE Loss vs Epoch')
plt.legend()
plt.show()

y_train_pred = network_overfit.predict(X_train)
train_err = calculate_prediction_error(y_train_pred, y_train)
y_test_pred = network_overfit.predict(X_test)
test_err = calculate_prediction_error(y_test_pred, y_test)

print('Training error = {0} and testing error = {1}'.format(train_err, test_err))

# %%
# 2b
# Weight initialization of randomly sampled standard gaussain 
r0 = 0.00007
d = 0.1 # Larger d means slower decay of r0, smaller d means faster decay of r0
T = 100
t_vals = np.arange(T)
r_sched = r0 / (1 + (r0/d)*t_vals)

H_vals = np.array([5, 10, 25, 50, 100])

for H in H_vals:
    network = NeuralNetwork(H)
    MSE_by_epoch = network.train(X_train, y_train, random_weight_initialization=True, r_sched=r_sched)
    plt.figure()
    plt.plot(MSE_by_epoch, label='Training Set MSE Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('MSE Loss vs Epoch H={0}'.format(H))
    plt.legend()
    plt.show()

    y_train_pred = network.predict(X_train)
    train_err = calculate_prediction_error(y_train_pred, y_train)
    y_test_pred = network.predict(X_test)
    test_err = calculate_prediction_error(y_test_pred, y_test)
    print('H = {0}, gamma_0 = {1}, d = {2}: training error = {3} and testing error = {4}'.format(H, r0, d, train_err, test_err))
# %%
# 2c
# Weight initialization of zero
r0 = 0.00007
d = 0.1 # Larger d means slower decay of r0, smaller d means faster decay of r0
T = 100
t_vals = np.arange(T)+1
r_sched = r0 / (1 + (r0/d)*t_vals)

H_vals = np.array([5, 10, 25, 50, 100])

for H in H_vals:
    network = NeuralNetwork(H)
    MSE_by_epoch = network.train(X_train, y_train, random_weight_initialization=False, r_sched=r_sched)
    plt.figure()
    plt.plot(MSE_by_epoch, label='Training Set MSE Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('MSE Loss vs Epoch H={0}'.format(H))
    plt.legend()
    plt.show()

    y_train_pred = network.predict(X_train)
    train_err = calculate_prediction_error(y_train_pred, y_train)
    y_test_pred = network.predict(X_test)
    test_err = calculate_prediction_error(y_test_pred, y_test)
    print('H = {0}, gamma_0 = {1}, d = {2}: training error = {3} and testing error = {4}'.format(H, r0, d, train_err, test_err))
