# %%
# Libraries 
import numpy as np
import pandas as pd

# Personal Libraries
from perceptron import Perceptron, VotedPerceptron, AveragedPerceptron, calculate_prediction_error

# %%
# General Functions
def convert_0_to_minus1(y):
    ycp = y.copy()
    ycp[ycp==0] = -1
    return ycp
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
perceptron_model = Perceptron()
w = perceptron_model.train(X_train, y_train, T=10, r=0.01)
y_train_pred = perceptron_model.predict(X_train)
y_test_pred = perceptron_model.predict(X_test)
y_train_pred_err = calculate_prediction_error(y_train_pred, y_train)
y_test_pred_err = calculate_prediction_error(y_test_pred, y_test)

print("Perceptron model training error = {0}, test error = {1}\n".format(y_train_pred_err, y_test_pred_err))

# %%
# 2b
voted_perceptron_model = VotedPerceptron()
w_vals, c_vals = voted_perceptron_model.train(X_train, y_train, T=10, r=0.01)
y_train_pred = voted_perceptron_model.predict(X_train)
y_test_pred = voted_perceptron_model.predict(X_test)
y_train_pred_err = calculate_prediction_error(y_train_pred, y_train)
y_test_pred_err = calculate_prediction_error(y_test_pred, y_test)

print("Voted perceptron model training error = {0}, test error = {1}\n".format(y_train_pred_err, y_test_pred_err))

# %%
# 2c
averaged_perceptron_model = AveragedPerceptron()
a = averaged_perceptron_model.train(X_train, y_train, T=10, r=0.01)
y_train_pred = averaged_perceptron_model.predict(X_train)
y_test_pred = averaged_perceptron_model.predict(X_test)
y_train_pred_err = calculate_prediction_error(y_train_pred, y_train)
y_test_pred_err = calculate_prediction_error(y_test_pred, y_test)

print("Averaged perceptron model training error = {0}, test error = {1}\n".format(y_train_pred_err, y_test_pred_err))
# %%

# %%
