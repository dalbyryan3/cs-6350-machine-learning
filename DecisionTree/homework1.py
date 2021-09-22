# %% 
from os import getpgid
from decision_tree import DecisionTree, Node


# %%
# Functions
def prediction_error(y_pred, y_actual):
    # Determine prediction error 
    if len(y_pred) != len(y_actual):
        raise Exception('y_pred and y_actual are not the same length')
    error_count = 0
    for i in range(len(y_pred)):
        if y_pred[i] != y_actual[i]:
            error_count +=1
    return error_count/len(y_pred)

def extract_ID3_input(filename, attributes):
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
                example_dict[a] = terms[i]  
            labels.append(terms[-1])
            S.append(example_dict)
    attribute_possible_vals_in_this_data = {}
    for a in attributes:
        attribute_possible_vals_in_this_data[a] = list(DecisionTree.get_attribute_values(S, a))
    return (S, attributes, labels, attribute_possible_vals_in_this_data)

# %%
# Decision tree with car data

# Extract S, attributes, labels, and a map of all possible values of each attribute
(S_car_train, attributes_car, labels_car_train, attribute_possible_vals_car) = extract_ID3_input('car/train.csv', ['buying','maint','doors','persons','lug_boot','safety'])
(S_car_test, _, labels_car_test, _) = extract_ID3_input('car/test.csv', ['buying','maint','doors','persons','lug_boot','safety'])

# Train a DecisionTree from car dataset
def get_car_prediction_error(metric, max_depth):
    dc_car = DecisionTree(attribute_possible_vals_car)
    _ = dc_car.train(S_car_train, attributes_car, labels_car_train, metric=metric, max_depth=max_depth)
    pred_labels_car_train = dc_car.predict(S_car_train)
    pred_err_car_train = prediction_error(pred_labels_car_train, labels_car_train)
    pred_labels_car_test = dc_car.predict(S_car_test)
    pred_err_car_test = prediction_error(pred_labels_car_test, labels_car_test)
    print('Max depth = {0}:'.format(max_depth))
    print('Train error = {0}'.format(pred_err_car_train))
    print('Test error = {0}'.format(pred_err_car_test))
    print()

print('Car data set decision tree error:')
metrics_tuple = [(DecisionTree.entropy, 'Entropy'), (DecisionTree.majority_error, 'Majority Error'), (DecisionTree.gini_index, 'Gini Index')]
for metric in metrics_tuple:
    print('{0}:\n'.format(metric[1]))
    for i in range(1,7):
        get_car_prediction_error(metric[0], i)
print()
print()

# %%
# Decision tree with bank data- unknown as attribute value


print('Bank data set decision tree error with unknown as an attribute value:')

# %%
# Decision tree with bank data- unknown as majority of the values of the same attribute

print('Bank data set decision tree error with unknown as the majority of the values of the same attribute:')