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

print("Perceptron model:")
print("Learned weight vector = {0}".format(w))
print("Training error = {0}, test error = {1}\n\n".format(y_train_pred_err, y_test_pred_err))

# %%
# 2b
voted_perceptron_model = VotedPerceptron()
w_vals, c_vals = voted_perceptron_model.train(X_train, y_train, T=10, r=0.01)
y_train_pred = voted_perceptron_model.predict(X_train)
y_test_pred = voted_perceptron_model.predict(X_test)
y_train_pred_err = calculate_prediction_error(y_train_pred, y_train)
y_test_pred_err = calculate_prediction_error(y_test_pred, y_test)

w_vals_list = [np.array2string(w_vals[i,:], formatter={'float_kind':'{0:.4f}'.format}) for i in range(w_vals.shape[0])]
table_df = pd.DataFrame({'Counts': c_vals, 'Weights': w_vals_list})
latex_table = table_df.to_latex(column_format='|c|c|', index=None, longtable=True, caption='Voted Perceptron Counts(Votes) and Weight Vectors', label='tab:2b')

print("Voted perceptron model:")
print("Learned weight vectors as latex table = \n\n{0}\n\n".format(latex_table))
print("Training error = {0}, test error = {1}\n\n".format(y_train_pred_err, y_test_pred_err))

# %%
# 2c
averaged_perceptron_model = AveragedPerceptron()
a = averaged_perceptron_model.train(X_train, y_train, T=10, r=0.01)
y_train_pred = averaged_perceptron_model.predict(X_train)
y_test_pred = averaged_perceptron_model.predict(X_test)
y_train_pred_err = calculate_prediction_error(y_train_pred, y_train)
y_test_pred_err = calculate_prediction_error(y_test_pred, y_test)

print("Averaged perceptron model:")
print("Learned weight vector (sum of all weights investigated, a) = {0}".format(a))
print("Training error = {0}, test error = {1}\n\n".format(y_train_pred_err, y_test_pred_err))