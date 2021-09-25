# %% 
from decision_tree import DecisionTree
import statistics

# %%
# General functions
def prediction_error(y_pred, y_actual):
    # Determine prediction error 
    if len(y_pred) != len(y_actual):
        raise Exception('y_pred and y_actual are not the same length')
    error_count = 0
    for i in range(len(y_pred)):
        if y_pred[i] != y_actual[i]:
            error_count +=1
    return error_count/len(y_pred)

def extract_ID3_input(filename, attributes, attribute_discretize_idx_to_thresh_map={}, unknown_value='unknown', unknown_replacement_map={}):
    S = []
    labels = []
    with open(filename, 'r') as f:
        for line in f:
            terms = line.strip().split(',')
            if len(terms) != (len(attributes)+1):
                raise Exception('Length of given attributes does not match parsed length of a line in filename to extract from (a line in filename should have the number of terms in attributes plus one (for the label))')
            # Make dictionary mapping from attribute to value
            example_dict = {}
            for i in range(len(attributes)):
                a = attributes[i]
                value = terms[i]
                if value == unknown_value and a in unknown_replacement_map:
                    value = unknown_replacement_map[a]
                if i in attribute_discretize_idx_to_thresh_map:
                    threshold = attribute_discretize_idx_to_thresh_map[i]
                    value = int(float(value)>threshold)
                example_dict[a] = value  
            labels.append(terms[-1])
            S.append(example_dict)
    attribute_possible_vals_in_this_data = {}
    for a in attributes:
        attribute_possible_vals_in_this_data[a] = list(DecisionTree.get_attribute_values(S, a))
    return (S, attributes, labels, attribute_possible_vals_in_this_data)

def get_attribute_discretize_idx_to_thresh_map(filename, attributes, attribute_idx_to_discretize):
    discretize_dict = {}
    with open(filename, 'r') as f:
        for line in f:
            terms = line.strip().split(',')
            if len(terms) != (len(attributes)+1):
                raise Exception('Length of given attributes does not match parsed length of a line in filename to extract from (a line in filename should have the number of terms in attributes plus one (for the label))')
            for i in range(len(attributes)):
                if i in attribute_idx_to_discretize:
                    value = float(terms[i])
                    if i in discretize_dict:
                        discretize_dict[i].append(value)
                    else:
                        discretize_dict[i] = [value]

    attribute_discretize_idx_to_thresh_map = {}
    for idx in attribute_idx_to_discretize:
        attribute_discretize_idx_to_thresh_map[idx] = statistics.median(discretize_dict[idx])
    return attribute_discretize_idx_to_thresh_map

def get_attribute_to_most_common_value_map(filename, attributes, value_to_ignore='unknown'):
    with open(filename, 'r') as f:
        values_dict = {}
        for line in f:
            terms = line.strip().split(',')
            if len(terms) != (len(attributes)+1):
                raise Exception('Length of given attributes does not match parsed length of a line in filename to extract from (a line in filename should have the number of terms in attributes plus one (for the label))')
            for i in range(len(attributes)):
                a = attributes[i]
                value = terms[i]
                if (value == 'unknown'):
                    continue
                if a in values_dict:
                    values_dict[a].append(value)
                else:
                    values_dict[a] = [value]
    attribute_to_most_common_value_map = {}
    for a in values_dict:
        a_values = values_dict[a]
        attribute_to_most_common_value_map[a] = max(set(a_values), key=a_values.count)
    return attribute_to_most_common_value_map

def get_prediction_error(metric, max_depth, attributes, attribute_possible_vals,  S_train, labels_train, S_test, labels_test, verbose=True):
    dc = DecisionTree(attribute_possible_vals)
    _ = dc.train(S_train, attributes, labels_train, metric=metric, max_depth=max_depth)
    pred_labels_train = dc.predict(S_train)
    pred_err_train = prediction_error(pred_labels_train, labels_train)
    pred_labels_test = dc.predict(S_test)
    pred_err_test = prediction_error(pred_labels_test, labels_test)
    if verbose:
        print('Max depth = {0}:'.format(max_depth))
        print('Train error = {0}'.format(pred_err_train))
        print('Test error = {0}'.format(pred_err_test))
        print()
    return (max_depth, pred_err_train, pred_err_test)

def errors_for_metrics(attributes, attribute_possible_vals, S_train, labels_train, S_test, labels_test, max_depth_top_range, verbose=True):
    metrics_tuple = [(DecisionTree.entropy, 'Entropy'), (DecisionTree.majority_error, 'Majority Error'), (DecisionTree.gini_index, 'Gini Index')]
    for metric in metrics_tuple:
        print('{0}:'.format(metric[1]))
        train_errors = []
        test_errors = []
        for i in range(1,max_depth_top_range+1):
            (_,train_error,test_error) = get_prediction_error(metric[0], i, attributes, attribute_possible_vals, S_train, labels_train, S_test, labels_test, verbose=verbose)
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
(S_car_train, attributes_car, labels_car_train, attribute_possible_vals_car) = extract_ID3_input('car/train.csv', ['buying','maint','doors','persons','lug_boot','safety'])
(S_car_test, _, labels_car_test, _) = extract_ID3_input('car/test.csv', ['buying','maint','doors','persons','lug_boot','safety'])

errors_for_metrics(attributes_car, attribute_possible_vals_car, S_car_train, labels_car_train, S_car_test, labels_car_test, 6, verbose=True)

# %%
# Decision tree with bank data- unknown as attribute value
# Extract S, attributes, labels, and a map of all possible values of each attribute
bank_attributes = ['age','job','marital','education','default','balance','housing','loan','contact','day','month','duration','campaign','pdays','previous','poutcome']
bank_attributes_idx_to_discretize = [0,5,9,11,12,13,14]

idx_thresh_map = get_attribute_discretize_idx_to_thresh_map('bank/train.csv', bank_attributes, bank_attributes_idx_to_discretize)

(S_bank_train, attributes_bank, labels_bank_train, attribute_possible_vals_bank) = extract_ID3_input('bank/train.csv', bank_attributes, attribute_discretize_idx_to_thresh_map=idx_thresh_map)
(S_bank_test, _, labels_bank_test, _) = extract_ID3_input('bank/test.csv', bank_attributes, attribute_discretize_idx_to_thresh_map=idx_thresh_map)

print('Bank data set decision tree error with unknown as an attribute value:')
errors_for_metrics(attributes_bank, attribute_possible_vals_bank, S_bank_train, labels_bank_train, S_bank_test, labels_bank_test, 16, verbose=True)

# %%
# Decision tree with bank data- unknown as majority of the values of the same attribute
replacement_map = get_attribute_to_most_common_value_map('bank/train.csv', bank_attributes)

(S_bank_train, attributes_bank, labels_bank_train, attribute_possible_vals_bank) = extract_ID3_input('bank/train.csv', bank_attributes, attribute_discretize_idx_to_thresh_map=idx_thresh_map, unknown_replacement_map=replacement_map)
(S_bank_test, _, labels_bank_test, _) = extract_ID3_input('bank/test.csv', bank_attributes, attribute_discretize_idx_to_thresh_map=idx_thresh_map, unknown_replacement_map=replacement_map)

print('Bank data set decision tree error with unknown as the majority of the values of the same attribute:')
errors_for_metrics(attributes_bank, attribute_possible_vals_bank, S_bank_train, labels_bank_train, S_bank_test, labels_bank_test, 16, verbose=True)