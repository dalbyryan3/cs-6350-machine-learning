# %% 
from os import getpgid
from decision_tree import DecisionTree, Node

# %%
# Extract S, attributes, labels, and a map of all possible values of each attribute
(S_car_train, attributes_car, labels_car_train, attribute_possible_vals_car) = DecisionTree.extract_ID3_input('car/train.csv', ['buying','maint','doors','persons','lug_boot','safety'])
(S_car_test, _, labels_car_test, _) = DecisionTree.extract_ID3_input('car/test.csv', ['buying','maint','doors','persons','lug_boot','safety'])

# %%
# Train a DecisionTree from car dataset
def get_car_prediction_error(metric, max_depth):
    dc_car = DecisionTree(attribute_possible_vals_car)
    _ = dc_car.train(S_car_train, attributes_car, labels_car_train, metric=metric, max_depth=max_depth)
    pred_labels_car_train = dc_car.predict(S_car_train)
    pred_err_car_train = DecisionTree.prediction_error(pred_labels_car_train, labels_car_train)
    pred_labels_car_test = dc_car.predict(S_car_test)
    pred_err_car_test = DecisionTree.prediction_error(pred_labels_car_test, labels_car_test)
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
