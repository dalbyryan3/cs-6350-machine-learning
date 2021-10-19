# %%
# Libraries
import sys
import numpy as np
import matplotlib.pyplot as plt
import pickle

# Personal libraries
from ensemble import AdaBoost, BaggedDecisionTree, RandomForest
sys.path.insert(1, '../')
from DecisionTree.decision_tree import DecisionTree

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

# 2a- Saving long-run data
with open('hw2_2a.pkl', 'wb') as f:
    pickle.dump([train_err_vals, test_err_vals, stump_train_err_vals, stump_test_err_vals, models], f)

# %%
# 2a- Loading long-run data
with open('hw2_2a.pkl', 'rb') as f:
    train_err_vals, test_err_vals, stump_train_err_vals, stump_test_err_vals, models = pickle.load(f)
# %%
# 2a- Plotting
# First figure
plt.figure()
plt.plot(train_err_vals, label='Training Error')
plt.plot(test_err_vals, label='Test Error')
plt.legend()
plt.title('AdaBoost Prediction Errors')
plt.ylabel('Error')
plt.xlabel('Number of Hypothesis Used (T)')
plt.show()

# Second figure
plt.figure()
plt.plot(stump_train_err_vals, '.', label='Individual Stump Training Error', linewidth=0.7)
plt.plot(stump_test_err_vals, '.', label='Individual Stump Test Error', linewidth=0.7)
plt.legend()
plt.title('AdaBoost Prediction Errors of Decision Stump Learned at T')
plt.ylabel('Error')
plt.xlabel('Number of Hypothesis Used (T)')
plt.show()





# %%
# 2b- LONG RUNNING
Tmax = 500
Tvals = list(range(1,Tmax+1))
train_err_vals = []
test_err_vals = []
model = BaggedDecisionTree(attribute_possible_vals_bank)
for T in Tvals:
    print('Number of bagged trees used for this model: {0}'.format(T))
    model.train_one_extra_tree(S_bank_train, attributes_bank, labels_bank_train)

    train_pred = model.predict(S_bank_train)
    train_err = DecisionTree.prediction_error(train_pred, labels_bank_train)
    train_err_vals.append(train_err)

    test_pred = model.predict(S_bank_test)
    test_err = DecisionTree.prediction_error(test_pred, labels_bank_test)
    test_err_vals.append(test_err)

# 2b- Saving long-run data
with open('hw2_2b.pkl', 'wb') as f:
    pickle.dump([train_err_vals, test_err_vals, model], f)

# %%
# 2b- Loading long-run data
with open('hw2_2b.pkl', 'rb') as f:
    train_err_vals, test_err_vals, model = pickle.load(f)

# %%
# 2b- Plotting
plt.figure()
plt.plot(train_err_vals, label='Training Error')
plt.plot(test_err_vals, label='Test Error')
plt.legend()
plt.title('Bagged Decision Tree Prediction Errors')
plt.ylabel('Error')
plt.xlabel('Number of Bagged Decision Trees Used (T)')
plt.show()




# %%
# 2c- LONG RUNNING
bagged_tree_list = []
single_tree_list = []
for i in range(100):
    print('Bagged Tree Number: {0}'.format(i+1))
    S_bank_train_sample, labels_bank_train_sample = BaggedDecisionTree.uniform_sample_with_replacement(S_bank_train, labels_bank_train, sample_size=1000)
    bagged_tree = BaggedDecisionTree(attribute_possible_vals_bank)
    bagged_tree.train(S_bank_train_sample, attributes_bank, labels_bank_train_sample, 500)
    bagged_tree_list.append(bagged_tree)
    single_tree_list.append(bagged_tree.decision_tree_list[0])

# 2c- Saving long-run data
with open('hw2_2c1.pkl', 'wb') as f:
    pickle.dump([bagged_tree_list, single_tree_list], f)

# %%
# 2c- Loading long-run data
with open('hw2_2c1.pkl', 'rb') as f:
    bagged_tree_list, single_tree_list = pickle.load(f)
# %% 
# 2c- Single tree bias, variance, and general squared error calculation
single_tree_mat = np.zeros((len(S_bank_test), len(single_tree_list)))
for i, single_tree in enumerate(single_tree_list):
    single_tree_pred = np.array(single_tree.predict(S_bank_test))
    single_tree_mat[:, i] = single_tree_pred

single_tree_pred_sum = np.sum(single_tree_mat, axis=1)
single_tree_pred_avg = single_tree_pred_sum / len(single_tree_list)
single_tree_bias = np.average((single_tree_pred_avg - np.array(labels_bank_test))**2)

single_tree_var = np.average(np.var(single_tree_mat, axis=1))

single_tree_sqerr = single_tree_bias + single_tree_var

# %% 
# 2c- Bagged tree bias, variance, and general squared error calculation - May run long
bagged_tree_mat = np.zeros((len(S_bank_test), len(bagged_tree_list)))
for i, bagged_tree in enumerate(bagged_tree_list):
    bagged_tree_pred = np.array(bagged_tree.predict(S_bank_test))
    bagged_tree_mat[:, i] = bagged_tree_pred

bagged_tree_pred_sum = np.sum(bagged_tree_mat, axis=1)
bagged_tree_pred_avg = bagged_tree_pred_sum / len(bagged_tree_list)
bagged_tree_bias = np.average((bagged_tree_pred_avg - np.array(labels_bank_test))**2)

bagged_tree_var = np.average(np.var(bagged_tree_mat, axis=1))

bagged_tree_sqerr = bagged_tree_bias + bagged_tree_var

# 2c- Saving long-run data 2
with open('hw2_2c2.pkl', 'wb') as f:
    pickle.dump([bagged_tree_bias, bagged_tree_var, bagged_tree_sqerr, bagged_tree_mat], f)

# %%
# 2c- Loading long-run data 2
with open('hw2_2c2.pkl', 'rb') as f:
    bagged_tree_bias, bagged_tree_var, bagged_tree_sqerr, bagged_tree_mat = pickle.load(f)

# %% 
# 2c- Print results
print('Single tree bias: {0:.3f}, variance: {1:.3f}, general squared error: {2:.3f}\n'.format(single_tree_bias, single_tree_var, single_tree_sqerr))

print('Bagged tree bias: {0:.3f}, variance: {1:.3f}, general squared error: {2:.3f}\n'.format(bagged_tree_bias, bagged_tree_var, bagged_tree_sqerr))




# %%
# 2d- LONG RUNNING
Tmax = 500 
Tvals = list(range(1,Tmax+1))
train_err_vals_dict = {2:[],4:[],6:[]}
test_err_vals_dict = {2:[],4:[],6:[]}
model_dict = {2:[],4:[],6:[]}

for sample_size in train_err_vals_dict.keys():
    train_err_vals = []
    test_err_vals = []
    model = RandomForest(attribute_possible_vals_bank)
    print()
    print('Now starting random forest with an attribute sample size {0}'.format(sample_size))
    for T in Tvals:
        print('Number of random forest decision trees used for this model: {0}'.format(T))
        model.train_one_extra_tree(S_bank_train, attributes_bank, labels_bank_train, random_forest_sampling_size=sample_size)

        train_pred = model.predict(S_bank_train)
        train_err = DecisionTree.prediction_error(train_pred, labels_bank_train)
        train_err_vals.append(train_err)

        test_pred = model.predict(S_bank_test)
        test_err = DecisionTree.prediction_error(test_pred, labels_bank_test)
        test_err_vals.append(test_err)
    train_err_vals_dict[sample_size] = train_err_vals
    test_err_vals_dict[sample_size] = test_err_vals
    model_dict[sample_size] = model

# 2d- Saving long-run data
with open('hw2_2d.pkl', 'wb') as f:
    pickle.dump([train_err_vals_dict, test_err_vals_dict, model_dict], f)

# %%
# 2d- Loading long-run data
with open('hw2_2d.pkl', 'rb') as f:
    train_err_vals_dict, test_err_vals_dict, model_dict = pickle.load(f)

# %%
# 2d- Plotting
for sample_size in train_err_vals_dict.keys():
    plt.figure()
    plt.plot(train_err_vals_dict[sample_size], label='Training Error')
    plt.plot(test_err_vals_dict[sample_size], label='Test Error')
    plt.legend()
    plt.title('Random Forest Prediction Errors for Attribute Sample Size of {0}'.format(sample_size))
    plt.ylabel('Error')
    plt.xlabel('Number of Random Forest Decision Trees Used (T)')
    plt.show()
plt.figure()
for sample_size in train_err_vals_dict.keys():
    plt.plot(train_err_vals_dict[sample_size], label='Training Error {0}'.format(sample_size))
    plt.plot(test_err_vals_dict[sample_size], label='Test Error {0}'.format(sample_size))
plt.legend(loc='right')
plt.title('Random Forest Prediction Errors for Various Attribute Sample Sizes')
plt.ylabel('Error')
plt.xlabel('Number of Random Forest Decision Trees Used (T)')
plt.show()
# %%
# 2e- LONG RUNNING
random_forest_list = []
single_tree_list = []
for i in range(100):
    print('Random Forest Number: {0}'.format(i+1))
    S_bank_train_sample, labels_bank_train_sample = RandomForest.uniform_sample_with_replacement(S_bank_train, labels_bank_train, sample_size=1000)
    random_forest = RandomForest(attribute_possible_vals_bank)
    random_forest.train(S_bank_train_sample, attributes_bank, labels_bank_train_sample, 500)
    random_forest_list.append(random_forest)
    single_tree_list.append(random_forest.decision_tree_list[0])

# 2e- Saving long-run data
with open('hw2_2e1.pkl', 'wb') as f:
    pickle.dump([random_forest_list, single_tree_list], f)

# %%
# 2e- Loading long-run data
with open('hw2_2e1.pkl', 'rb') as f:
    random_forest_list, single_tree_list = pickle.load(f)
# %% 
# 2e- Single tree bias, variance, and general squared error calculation
single_tree_mat = np.zeros((len(S_bank_test), len(single_tree_list)))
for i, single_tree in enumerate(single_tree_list):
    single_tree_pred = np.array(single_tree.predict(S_bank_test))
    single_tree_mat[:, i] = single_tree_pred

single_tree_pred_sum = np.sum(single_tree_mat, axis=1)
single_tree_pred_avg = single_tree_pred_sum / len(single_tree_list)
single_tree_bias = np.average((single_tree_pred_avg - np.array(labels_bank_test))**2)

single_tree_var = np.average(np.var(single_tree_mat, axis=1))

single_tree_sqerr = single_tree_bias + single_tree_var

# %% 
# 2e- Random forest bias, variance, and general squared error calculation - May run long
random_forest_mat = np.zeros((len(S_bank_test), len(random_forest_list)))
for i, random_forest in enumerate(random_forest_list):
    random_forest_pred = np.array(random_forest.predict(S_bank_test))
    random_forest_mat[:, i] = random_forest_pred

random_forest_pred_sum = np.sum(random_forest_mat, axis=1)
random_forest_pred_avg = random_forest_pred_sum / len(random_forest_list)
random_forest_bias = np.average((random_forest_pred_avg - np.array(labels_bank_test))**2)

random_forest_var = np.average(np.var(random_forest_mat, axis=1))

random_forest_sqerr = random_forest_bias + random_forest_var

# 2e- Saving long-run data 2
with open('hw2_2e2.pkl', 'wb') as f:
    pickle.dump([random_forest_bias, random_forest_var, random_forest_sqerr, random_forest_mat], f)

# %%
# 2e- Loading long-run data 2
with open('hw2_2e2.pkl', 'rb') as f:
    random_forest_bias, random_forest_var, random_forest_sqerr, random_forest_mat = pickle.load(f)

# %% 
# 2e- Print results
print('Single tree bias: {0:.3f}, variance: {1:.3f}, general squared error: {2:.3f}\n'.format(single_tree_bias, single_tree_var, single_tree_sqerr))

print('Random forest bias: {0:.3f}, variance: {1:.3f}, general squared error: {2:.3f}\n'.format(random_forest_bias, random_forest_var, random_forest_sqerr))