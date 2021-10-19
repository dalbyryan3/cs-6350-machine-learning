# %% 
# Libraries
import statistics

# Personal Libraries 
from decision_tree import DecisionTree

# %%
# General functions

def hw1_get_prediction_error(metric, max_depth, attributes, attribute_possible_vals,  S_train, labels_train, S_test, labels_test, verbose=True):
    # Will train an ID3 model, give the max depth trained on, and prediction error on the train and test set
    dc = DecisionTree(attribute_possible_vals)
    _ = dc.train(S_train, attributes, labels_train, metric=metric, max_depth=max_depth)
    pred_labels_train = dc.predict(S_train)
    pred_err_train = DecisionTree.prediction_error(pred_labels_train, labels_train)
    pred_labels_test = dc.predict(S_test)
    pred_err_test = DecisionTree.prediction_error(pred_labels_test, labels_test)
    if verbose:
        print('Max depth = {0}:'.format(max_depth))
        print('Train error = {0}'.format(pred_err_train))
        print('Test error = {0}'.format(pred_err_test))
        print()
    return (max_depth, pred_err_train, pred_err_test)

def hw1_errors_for_metrics(attributes, attribute_possible_vals, S_train, labels_train, S_test, labels_test, max_depth_top_range, verbose=True):
    # Gets metrics for training with different heuristics by printing
    metrics_tuple = [(DecisionTree.entropy, 'Entropy'), (DecisionTree.majority_error, 'Majority Error'), (DecisionTree.gini_index, 'Gini Index')]
    for metric in metrics_tuple:
        print('{0}:'.format(metric[1]))
        train_errors = []
        test_errors = []
        for i in range(1,max_depth_top_range+1):
            (_,train_error,test_error) = hw1_get_prediction_error(metric[0], i, attributes, attribute_possible_vals, S_train, labels_train, S_test, labels_test, verbose=verbose)
            train_errors.append(train_error)
            test_errors.append(test_error)
        print('Average train error over max depth 1 to {0}: {1}'.format(max_depth_top_range, statistics.mean(train_errors)))
        print('Average test error over max depth 1 to {0}: {1}'.format(max_depth_top_range, statistics.mean(test_errors)))
        print()
    print()
    print()

# %%
# Decision tree with car data

# Extract S, attributes, labels, and a map of all possible values of each attribute
(S_car_train, attributes_car, labels_car_train, attribute_possible_vals_car) = DecisionTree.extract_ID3_input('car/train.csv', ['buying','maint','doors','persons','lug_boot','safety'])
(S_car_test, _, labels_car_test, _) = DecisionTree.extract_ID3_input('car/test.csv', ['buying','maint','doors','persons','lug_boot','safety'])

hw1_errors_for_metrics(attributes_car, attribute_possible_vals_car, S_car_train, labels_car_train, S_car_test, labels_car_test, 6, verbose=True)

# %%
# Decision tree with bank data- unknown as attribute value
# Extract S, attributes, labels, and a map of all possible values of each attribute
bank_attributes = ['age','job','marital','education','default','balance','housing','loan','contact','day','month','duration','campaign','pdays','previous','poutcome']
bank_attributes_idx_to_discretize = [0,5,9,11,12,13,14]

idx_thresh_map = DecisionTree.get_attribute_discretize_idx_to_thresh_map('bank/train.csv', bank_attributes, bank_attributes_idx_to_discretize)

(S_bank_train, attributes_bank, labels_bank_train, attribute_possible_vals_bank) = DecisionTree.extract_ID3_input('bank/train.csv', bank_attributes, attribute_discretize_idx_to_thresh_map=idx_thresh_map)
(S_bank_test, _, labels_bank_test, _) = DecisionTree.extract_ID3_input('bank/test.csv', bank_attributes, attribute_discretize_idx_to_thresh_map=idx_thresh_map)

print('Bank data set decision tree error with unknown as an attribute value:')
hw1_errors_for_metrics(attributes_bank, attribute_possible_vals_bank, S_bank_train, labels_bank_train, S_bank_test, labels_bank_test, 16, verbose=True)

# %%
# Decision tree with bank data- unknown as majority of the values of the same attribute
replacement_map = DecisionTree.get_attribute_to_most_common_value_map('bank/train.csv', bank_attributes)

(S_bank_train, attributes_bank, labels_bank_train, attribute_possible_vals_bank) = DecisionTree.extract_ID3_input('bank/train.csv', bank_attributes, attribute_discretize_idx_to_thresh_map=idx_thresh_map, unknown_replacement_map=replacement_map)
(S_bank_test, _, labels_bank_test, _) = DecisionTree.extract_ID3_input('bank/test.csv', bank_attributes, attribute_discretize_idx_to_thresh_map=idx_thresh_map, unknown_replacement_map=replacement_map)

print('Bank data set decision tree error with unknown as the majority of the values of the same attribute:')
hw1_errors_for_metrics(attributes_bank, attribute_possible_vals_bank, S_bank_train, labels_bank_train, S_bank_test, labels_bank_test, 16, verbose=True)