import math
from os import name
# car_test_data = pd.read_csv('car/test.csv', )
# car_train_data = pd.read_csv('car/train.csv')

# bank_test_data = pd.read_csv('bank/test.csv')
# bank_train_data = pd.read_csv('bank/train.csv')
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
    return (S, attributes, labels)

# (S, attributes, labels) = extract_ID3_input('car/test.csv', ['1','2','3','4','5','6'])

class Node:
    def __init__(self, name):
        self.name = name
        self.parent = None
        self.children = []
        self.weights = [] 
    def add_child(self, child, weight):
        self.add_children([child], [weight])
    def add_children(self, children_to_add, weights):
        if children_to_add is not None and len(children_to_add) == len(weights):
            for i in range(len(children_to_add)):
                child = children_to_add[i]
                weight = weights[i]
                assert isinstance(child, Node)
                self.children.append(child)
                self.weights.append(weight)
                child.parent = self
    def node_depth(self):
        depth = 0
        current_node = self
        while (current_node.parent is not None):
            depth += 1
            current_node = current_node.parent
        return depth

class DecisionTree:
    def __init__(self, attribute_possible_vals):
        # A dictionary mapping from keys of all potential attributes to a list of all possible values for that attribute

        self.attribute_possible_vals = attribute_possible_vals
        # self.root = Node('root')
        # self.current_node = self.root

    def ID3_train(self, S, attributes, labels, metric=None, max_depth=None):
        if len(S) <= 0 or len(labels) != len(S):
            raise Exception('Must have at least a single example and label and they must be the be same length')

        # All examples have same label:
        if(labels.count(labels[0]) == len(labels)):
            return Node(labels[0])# leaf node with the label
        # Attributes empty:
        if(len(attributes) <= 0):
            possible_labels = set(labels)
            most_common_label = max(possible_labels, key=labels.count)
            return Node(most_common_label)
        # Otherwise:
        (best_attribute, Sv_dict, _) = self.find_best_attribute(S, labels, attributes, metric=metric)
        root_node = Node(best_attribute)
        current_depth = self.number_of_attribute_splits(attributes)
        stop_tree_growth = False
        if(max_depth is not None and current_depth+1 >= max_depth):
            # If next depth is max depth then we set flag to stop tree growth
            stop_tree_growth = True
        for value in self.attribute_possible_vals[best_attribute]:
            if (value not in Sv_dict):
                possible_labels = set(labels)
                most_common_label = max(possible_labels, key=labels.count)
                root_node.add_child(Node(most_common_label), value)
            else:
                Sv_tuple = Sv_dict[value]
                Sv = Sv_tuple[0]
                labels_v = Sv_tuple[1]
                if (stop_tree_growth):
                    # To stop tree growth we set the labels to the most common value so next recursion returns Node(most common value)
                    possible_labels = set(labels_v)
                    most_common_label = max(possible_labels, key=labels_v.count)
                    labels_v = [most_common_label] * len(labels_v)
                reduced_attributes = set(attributes) - set([best_attribute])
                root_node.add_child(self.ID3_train(Sv, reduced_attributes, labels_v, metric=metric, max_depth=max_depth), value)
        return root_node
    
    def number_of_attribute_splits(self, current_attributes):
        return len(self.attribute_possible_vals.keys()) - len(current_attributes)

    @classmethod
    def entropy(cls, labels):
        N = len(labels)
        possible_labels = set(labels)
        entropy = 0
        for label in possible_labels:
            label_count = labels.count(label)
            label_prob = label_count/N
            if(label_prob != 0):
                entropy += -label_prob*math.log(label_prob)
        return entropy

    @classmethod
    def majority_error(cls, labels):
        N = len(labels)
        possible_labels = set(labels)
        most_common_label = max(possible_labels, key=labels.count)
        non_most_common_count = 0
        non_most_common_labels = possible_labels - {most_common_label}
        for label in non_most_common_labels:
            non_most_common_count += labels.count(label)
        return non_most_common_count/N

    @classmethod
    def gini_index(cls, labels):
        N = len(labels)
        possible_labels = set(labels)
        label_prob_sum = 0
        for label in possible_labels:
            label_count = labels.count(label)
            label_prob = label_count/N
            label_prob_sum += label_prob**2 
        return 1-label_prob_sum

    @classmethod
    def gain(cls, S, labels, attribute, metric=None):
        if metric is None:
            metric = cls.entropy
        N = len(S)
        S_metric = metric(labels)
        expected_metric_sum = 0
        Sv_dict = cls.get_value_subset(S, labels, attribute)
        for Sv_key in Sv_dict.keys():
            (Sv, labels_v) = Sv_dict[Sv_key]
            expected_metric_sum += (len(Sv)/N) * metric(labels_v)
        # Returns gain value and a dictionary that given a value of the attribute maps to a tuple of (Sv, labels_v) representing the subset of S all containing the same value of the attribute
        return (S_metric - expected_metric_sum, Sv_dict)

    @classmethod
    def get_attribute_values(cls, S, attribute):
        # Should likely just use get_value_subset and look at keys but this may be faster if subset isn't needed
        attribute_vals = set()
        for example in S:
            ex_val = example[attribute]
            attribute_vals.add(ex_val)
        return attribute_vals

    @classmethod
    def get_value_subset(cls, S, labels, attribute):
        if (len(S) != len(labels)):
            raise Exception('S must be the same length as corresponding labels')
        Sv_dict = {}
        for i in range(len(S)):
            example = S[i]
            label = labels[i]
            ex_val = example[attribute]
            if ex_val in Sv_dict:
                Sv_dict[ex_val][0].append(example)
                Sv_dict[ex_val][1].append(label)
            else:
                Sv_dict[ex_val] = ([example],[label])
        # a dictionary that given a value of the attribute maps to a tuple of (Sv, labels_v) representing the subset of S all containing the same value of the attribute
        return Sv_dict

    @classmethod
    def find_best_attribute(cls, S, labels, attributes, metric=None):
        if metric is None:
            metric = cls.entropy
        best_Sv_dict = None 
        best_attribute = None
        best_gain = 0
        for current_attribute in attributes:
            (current_gain, current_Sv_dict) = cls.gain(S, labels, current_attribute, metric=metric)
            # Will choose best gain based on this equality, if multiple gains of same value will arbitrarily choose one
            if(current_gain > best_gain):
                best_Sv_dict = current_Sv_dict
                best_attribute = current_attribute
                best_gain = current_gain
        # Returns a tuple of the best attribute, a dictionary maps to a tuple of (Sv, labels_v) representing the subsets of S for splitting on the best attribute, and the best gain value
        return (best_attribute, best_Sv_dict, best_gain)