from decision_tree import DecisionTree, Node

(S_train, attributes_train, labels_train, attribute_possible_vals_train) = DecisionTree.extract_ID3_input('car/train.csv', ['buying','maint','doors','persons','lug_boot','safety'])
(S_test, attributes_test, labels_test, attribute_possible_vals_test) = DecisionTree.extract_ID3_input('car/test.csv', ['buying','maint','doors','persons','lug_boot','safety'])

dc = DecisionTree(attribute_possible_vals_train)
dc.train(S_train, attributes_train, labels_train)
root = dc.visualize_tree(should_print=True)