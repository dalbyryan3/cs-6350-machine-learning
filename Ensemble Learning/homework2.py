# %%
from ensemble import AdaBoost
import sys
sys.path.insert(1, '../')
from DecisionTree.decision_tree import DecisionTree
import numpy as np
import matplotlib.pyplot as plt
import pickle

# %%
# General functions
def hw2_convert_labels(labels):
    convert_func = lambda label: 1.0 if label == 'yes' else -1.0
    return [convert_func(i) for i in labels]

# %%
# Decision tree with bank data- unknown as attribute value
# Extract S, attributes, labels, and a map of all possible values of each attribute
bank_attributes = ['age','job','marital','education','default','balance','housing','loan','contact','day','month','duration','campaign','pdays','previous','poutcome']
bank_attributes_idx_to_discretize = [0,5,9,11,12,13,14]

idx_thresh_map = DecisionTree.get_attribute_discretize_idx_to_thresh_map('bank/train.csv', bank_attributes, bank_attributes_idx_to_discretize)

(S_bank_train, attributes_bank, labels_bank_train, attribute_possible_vals_bank) = DecisionTree.extract_ID3_input('bank/train.csv', bank_attributes, attribute_discretize_idx_to_thresh_map=idx_thresh_map)
(S_bank_test, _, labels_bank_test, _) = DecisionTree.extract_ID3_input('bank/test.csv', bank_attributes, attribute_discretize_idx_to_thresh_map=idx_thresh_map)

# Convert labels:
labels_bank_train = hw2_convert_labels(labels_bank_train)
labels_bank_test = hw2_convert_labels(labels_bank_test)

# # %%
# T = 10
# print('Number of boosting training epochs used for this model: {0}'.format(T))
# model = AdaBoost(attribute_possible_vals_bank)
# model.train(S_bank_train, attributes_bank, labels_bank_train, T)
# train_pred = model.predict(S_bank_train)
# test_pred = model.predict(S_bank_test)
# train_err = DecisionTree.prediction_error(train_pred, labels_bank_train)
# test_err = DecisionTree.prediction_error(test_pred, labels_bank_test)

# %% 
# 2a- LONG RUNNING
Tmax = 500 
Tvals = list(range(1,Tmax+1))
train_err_vals = []
test_err_vals = []
stump_train_err_vals = []
stump_test_err_vals = []
models = []
for T in Tvals:
    print('Number of boosting training epochs used for this model: {0}'.format(T))
    model = AdaBoost(attribute_possible_vals_bank)
    model.train(S_bank_train, attributes_bank, labels_bank_train, T)
    models.append(model)

    train_pred = model.predict(S_bank_train)
    train_err = DecisionTree.prediction_error(train_pred, labels_bank_train)
    train_err_vals.append(train_err)

    test_pred = model.predict(S_bank_test)
    test_err = DecisionTree.prediction_error(test_pred, labels_bank_test)
    test_err_vals.append(test_err)
    
    (stump, _) = model.stump_alpha_t_list[-1]

    stump_train_pred = stump.predict(S_bank_train)
    stump_train_err = DecisionTree.prediction_error(stump_train_pred, labels_bank_train)
    stump_train_err_vals.append(stump_train_err)

    stump_test_pred = stump.predict(S_bank_test)
    stump_test_err = DecisionTree.prediction_error(stump_test_pred, labels_bank_test)
    stump_test_err_vals.append(stump_test_err)

with open('objs.pkl', 'wb') as f:
    pickle.dump([train_err_vals, test_err_vals, stump_train_err_vals, stump_test_err_vals], f)

# %%
# 2a- Plotting
with open('objs.pkl', 'rb') as f:
    train_err_vals, test_err_vals, stump_train_err_vals, stump_test_err_vals = pickle.load(f)
# First figure
plt.figure()
plt.plot(train_err_vals, label='Training Error')
plt.plot(test_err_vals, label='Test Error')
plt.legend()
plt.title('Adaboost Prediction Errors')
plt.ylabel('Error')
plt.xlabel('Number of Hypothesis Used (T)')
plt.show()

# Second figure
plt.figure()
plt.plot(stump_train_err_vals, label='Individual Stump Training Error')
plt.plot(stump_test_err_vals, label='Individual Stump Test Error')
plt.legend()
plt.title('Adaboost Prediction Errors of Decision Stump Learned at T')
plt.ylabel('Error')
plt.xlabel('Number of Hypothesis Used (T)')
plt.show()
# %%

# %%
