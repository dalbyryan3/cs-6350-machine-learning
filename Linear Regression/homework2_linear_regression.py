# %%
# Libraries 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Personal Libraries
from linear_regression import LMSRegression

# %%
# Load data
data_root_path = './concrete'
train_data = pd.read_csv('{0}/train.csv'.format(data_root_path), header=None) 
test_data = pd.read_csv('{0}/test.csv'.format(data_root_path), header=None) 
X_train = train_data.iloc[:, 0:7].to_numpy()
y_train = train_data.iloc[:, 7].to_numpy()
X_test = test_data.iloc[:, 0:7].to_numpy()
y_test = test_data.iloc[:, 7].to_numpy()

# %%
# 4a- Batch gradient descent
batch_LMS_model = LMSRegression()
learning_rate = 0.01
norm_w_diff_thresh = 1e-6
cost_vals = batch_LMS_model.train_batch_gradient_descent(X_train, y_train, r=learning_rate, norm_w_diff_thresh=norm_w_diff_thresh)
batch_final_w = batch_LMS_model.w
batch_test_data_cost = LMSRegression.cost(X_test, y_test, batch_final_w)
# %%
# 4a- Plot cost at each step
plt.figure()
plt.plot(cost_vals, label='Training Cost')
plt.title('Batch Gradient Descent LMS Cost versus Epoch')
plt.xlabel('Number of Full Batch Gradient Steps (Epoch)')
plt.ylabel('LMS Cost')
plt.legend()
plt.show()
# %%
# 4a- Display final weight vector and final cost on training data
print('After training using batch gradient descent and a learning rate of r = {0} the final weight vector was {1}.'.format(learning_rate, batch_final_w))
print('The cost using the learned weight vector on the test data was {0:.3f}.'.format(batch_test_data_cost))

# %%
# 4b- Stochastic gradient descent
stochastic_LMS_model = LMSRegression()
learning_rate = 0.01
abs_cost_diff_thresh = 1e-6
cost_vals = stochastic_LMS_model.train_stochastic_gradient_descent(X_train, y_train, r=learning_rate, abs_cost_diff_thresh=abs_cost_diff_thresh)
stochastic_final_w = stochastic_LMS_model.w
stochastic_test_data_cost = LMSRegression.cost(X_test, y_test, stochastic_final_w)
# %%
# 4b- Plot cost at each step
plt.figure()
plt.plot(cost_vals, label='Training Cost')
plt.title('Stochastic Gradient Descent LMS Cost versus Updates')
plt.xlabel('Number of Stochastic Weight Updates')
plt.ylabel('LMS Cost')
plt.legend()
plt.show()
# %%
# 4b- Display final weight vector and final cost on training data
print('After training using stochastic gradient descent and a learning rate of r = {0} the final weight vector was {1}.'.format(learning_rate, stochastic_final_w))
print('The cost using the learned weight vector on the test data was {0:.3f}.'.format(stochastic_test_data_cost))

# %%
# 4c- Analytical weight vector solution
analytical_LMS_model = LMSRegression()
analytical_LMS_model.train_analytical(X_train, y_train)
analytical_final_w = analytical_LMS_model.w
analytical_test_data_cost = LMSRegression.cost(X_test, y_test, analytical_final_w)
# %%
# 4c- Display final weight vector and final cost on training data
print('After training using the direct analytical weight vector solution for LMS the final weight vector was {0}.'.format(analytical_final_w))
print('The cost using the learned weight vector on the test data was {0:.3f}.'.format(analytical_test_data_cost))

# %% 
# Weight vector comparison
print()
print('Batch w: {0}, cost: {1}\n Stochastic w: {2}, cost: {3}\n Analytical w: {4}, cost: {5}\n'.format(batch_final_w, batch_test_data_cost, stochastic_final_w, stochastic_test_data_cost, analytical_final_w, analytical_test_data_cost))
