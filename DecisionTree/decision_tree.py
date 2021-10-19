import math
import random 
import statistics

class Node:
    def __init__(self, name):
        """ Create a Node that can be used to build trees. A Node can have multiple children and each child has an associated weight that defines the path connecting this Node to the child Node. 
        i.e. These Nodes can create decision trees by having the name specify an attribute and the weights specifying a value of that attribute. A leaf node would have name that would be interpreted as a label.

        Args:
            name (object): Name that defines this node  
        """
        self.name = name
        self.parent = None
        self.children = {} 
    def add_child(self, child, weight):
        self.add_children([child], [weight])
    def add_children(self, children_to_add, weights):
        if children_to_add is not None and len(children_to_add) == len(weights):
            for i in range(len(children_to_add)):
                child = children_to_add[i]
                weight = weights[i]
                assert isinstance(child, Node)
                self.children[weight] = child
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
        """ Creates DecisionTree object with a description of possible attributes and their possible values

        Args:
            attribute_possible_vals (dict of str: list[str]): A dictionary mapping from all possible attributes to a list containing every possible value for the attribute
        """
        self.attribute_possible_vals = attribute_possible_vals
        self.root = None

    def train(self, S, attributes, labels, metric=None, max_depth=None, weights=None, random_forest_attribute_sampling=False, random_forest_sampling_size=None):
        """ Trains decision tree on labelled data

        Args:
            S (list of dict[str: str]): A list with each element being a dictionary which represents an example. This dictionary maps from each possible attribute to the value of the attribute for the example. 
            attributes (list[str]): A list containing each possible attribute value 
            labels (list): A list of the labels for each example in S. Must be same order and length as S. 
            metric (Callable[[list], float], optional): Callable function representing the metric (heuristic) to use to determine how to split decision tree when training. Defaults to None which will use entropy as the heuristic.
            max_depth (int, optional): Maximum depth of decision tree, if examples not perfectly split and max_depth is hit will choose most common class as leaf node. Defaults to None which means tree depth will not be limited.
            weights (list[float], optional): Weights (probabilities) associated with each training example. Must be same length as labels and values should sum to 1. Defaults to None which means weights will be equal for all examples.
            random_forest_attribute_sampling (bool, optional): If random forest based attribute subsampling should be used for each split. This means only a "small" subset of attributes at a split are considered as candidates to split on. Defaults to False which means all attributes are considered for each split.
            random_forest_sampling_size (int, optional): If random forest based attribute subsampling is indicated to be used used for each split will try to use sample this subset size of attributes (if less attributes than number gives will use all attributes). Defaults to None which means the ceiling of the current attribute size / 5 is used (only if random forest based attribute subsampling is indicated to be used used).

        Returns:
            Node: Root node of the Decision tree. 
        """
        # If weights are None then set to be equal for all examples
        N = len(S)
        if weights is None: # If no weights for examples are given, weight each equally
           weights = [1/N] * N 
        self.root = self.ID3(S, attributes, labels, weights, metric=metric, max_depth=max_depth, random_forest_attribute_sampling=random_forest_attribute_sampling, random_forest_sampling_size=random_forest_sampling_size)
        return self.root
    
    def predict(self, S):
        """ Predicts on already trained decision tree

        Args:
            S (list of dict): A list with each element being a dictionary which represents an example. This dictionary maps from each possible attribute to the value of the attribute for the example. 

        Returns:
            List of predicted labels corresponding to each example given in S.
        """
        if self.root is None:
            raise Exception('DecisionTree must be trained on data')
        labels = []
        for example in S:
            current_node = self.root
            while len(current_node.children) >= 0:
                if len(current_node.children) == 0:
                    labels.append(current_node.name)
                    break
                else:
                    if(current_node.name not in example):
                        raise Exception('Example did not have attribute={0}'.format(current_node.name))
                    example_val = example[current_node.name]
                    if(example_val not in current_node.children):
                        raise Exception('There was no branch of attribute={0} with value={1}'.format(current_node.name, example_val))
                    current_node = current_node.children[example_val]
        return labels
        
    def ID3(self, S, attributes, labels, weights, metric=None, max_depth=None, random_forest_attribute_sampling = False, random_forest_sampling_size=None):
        # Standard ID3 algorithm
        if len(S) <= 0 or len(labels) != len(S):
            raise Exception('Must have at least a single example and label and they must be the be same length')

        # All examples have same label:
        if(labels.count(labels[0]) == len(labels)):
            return Node(labels[0]) # leaf node with the label

        # Attributes empty:
        if(len(attributes) <= 0):
            most_common_label = DecisionTree._find_most_weighted_label(labels, weights)
            return Node(most_common_label)

        # Select attributes if specified 
        selected_attributes = attributes
        if(random_forest_attribute_sampling):
            if random_forest_sampling_size is None: 
                G = math.ceil(len(attributes)/5) # Will consider dividing by 5 and rounding up a "sufficiently" small subset of the attributes 
            else:
                G = random_forest_sampling_size if random_forest_sampling_size < len(attributes) else len(attributes)
            selected_attributes = random.sample(attributes, G)

        # If examples have different labels and attributes are not empty:
        (best_attribute, Sv_dict, _) = self.find_best_attribute(S, labels, selected_attributes, weights, metric=metric)
        root_node = Node(best_attribute)
        current_depth = self.number_of_attribute_splits(attributes)

        # If next depth is max depth then we set flag to stop tree growth
        stop_tree_growth = False
        if(max_depth is not None and current_depth+1 >= max_depth):
            stop_tree_growth = True 

        for value in self.attribute_possible_vals[best_attribute]:
            if (value not in Sv_dict):
                most_common_label = DecisionTree._find_most_weighted_label(labels, weights) 
                root_node.add_child(Node(most_common_label), value)
            else:
                Sv_tuple = Sv_dict[value]
                Sv = Sv_tuple[0]
                labels_v = Sv_tuple[1]
                weights_v = Sv_tuple[2]
                if (stop_tree_growth):
                    # To stop tree growth we set the labels to the most common value so next recursion returns Node(most common value)
                    most_common_label = DecisionTree._find_most_weighted_label(labels_v, weights_v)
                    labels_v = [most_common_label] * len(labels_v)
                reduced_attributes = set(attributes) - set([best_attribute])
                root_node.add_child(self.ID3(Sv, reduced_attributes, labels_v, weights_v, metric=metric, max_depth=max_depth), value)
        return root_node
    
    def number_of_attribute_splits(self, current_attributes):
        return len(self.attribute_possible_vals.keys()) - len(current_attributes)

    def visualize_tree(self, should_print=False):
        tree_str = {} 
        # Essentially BFS
        queue = []
        visited = []
        queue.append((self.root,''))
        visited.append(self.root)
        while queue:
            current_node_tuple = queue.pop(0)
            current_node = current_node_tuple[0]
            current_node_name = str(current_node.name)
            current_node_parent_name = ''
            if (current_node.parent is not None):
                current_node_parent_name = str(current_node.parent.name)
            current_node_weight = str(current_node_tuple[1])
            current_depth = current_node.node_depth()
            if current_depth in tree_str:
                tree_str[current_depth] += current_node_parent_name + '--' + current_node_weight + '-->' + current_node_name + '    '
            else:
                tree_str[current_depth] = current_node_parent_name + '--' + current_node_weight + '-->' + current_node_name + '    '
            for weight in current_node.children.keys():
                child = current_node.children[weight]
                if(child not in visited):
                    queue.append((child, weight))
                    visited.append(child)
        if (should_print):
            print()
            print('Visualize tree: ')
            for key in tree_str.keys():
                print('DEPTH=' + str(key) + ': ' + tree_str[key])
                print()
                print()
            print('End visualization ')
            print()
        return tree_str

    @classmethod
    def _find_most_weighted_label(cls, labels, weights):
        weighted_sum_dict = {}
        for i, w in enumerate(weights):
            label = labels[i]
            if label in weighted_sum_dict:
                weighted_sum_dict[label] += w
            else:
                weighted_sum_dict[label] = w
        most_common_label = max(weighted_sum_dict, key=weighted_sum_dict.get)
        return most_common_label

    @classmethod
    def _normalize_weights(cls, weights):
        norm_const = 1.0 / sum(weights)
        norm_weights = [i * norm_const for i in weights]
        return norm_weights

    @classmethod
    def entropy(cls, labels, weights):
        N = len(labels)
        if len(weights) != len(labels):
            raise Exception('Weights must be same length as labels. Weights length: {0}, labels length: {1}'.format(len(weights), len(labels)))
        possible_labels = set(labels)
        entropy = 0
        # Note will normalize weights so it is possible to split weighted examples into subsets
        norm_weights = cls._normalize_weights(weights)
        for label in possible_labels:
            label_weights = [w for i, w in enumerate(norm_weights) if labels[i] == label]
            label_prob = sum(label_weights) 
            if(label_prob != 0):
                entropy += -label_prob*math.log(label_prob)
        return entropy

    @classmethod
    def majority_error(cls, labels, weights):
        N = len(labels)
        if len(weights) != len(labels):
            raise Exception('Weights must be same length as labels. Weights length: {0}, labels length: {1}'.format(len(weights), len(labels)))
        possible_labels = set(labels)
        most_common_label = max(possible_labels, key=labels.count)
        non_most_common_labels = possible_labels - {most_common_label}
        majority_err = 0
        # Note will normalize weights so it is possible to split weighted examples into subsets
        norm_weights = cls._normalize_weights(weights)
        for label in non_most_common_labels:
            label_weights = [w for i, w in enumerate(norm_weights) if labels[i] == label]
            majority_err += sum(label_weights)
        return majority_err 

    @classmethod
    def gini_index(cls, labels, weights):
        N = len(labels)
        if len(weights) != len(labels):
            raise Exception('Weights must be same length as labels. Weights length: {0}, labels length: {1}'.format(len(weights), len(labels)))
        possible_labels = set(labels)
        label_prob_sum = 0
        # Note will normalize weights so it is possible to split weighted examples into subsets
        norm_weights = cls._normalize_weights(weights)
        for label in possible_labels:
            label_weights = [w for i, w in enumerate(norm_weights) if labels[i] == label]
            label_prob = sum(label_weights) 
            label_prob_sum += label_prob**2 
        return 1-label_prob_sum

    @classmethod
    def gain(cls, S, labels, attribute, weights, metric=None):
        N = len(S)
        if metric is None:
            metric = cls.entropy
        S_metric = metric(labels, weights)
        expected_metric_sum = 0
        Sv_dict = cls.get_value_subset(S, labels, attribute, weights)
        for Sv_key in Sv_dict.keys(): # Loop through all examples by value (each Sv_key is a different value of attribute)
            (Sv, labels_v, weights_v) = Sv_dict[Sv_key]
            expected_metric_sum += (len(Sv)/N) * metric(labels_v, weights_v)
        # Returns gain value and a dictionary that given a value of the attribute maps to a tuple of (Sv, labels_v) representing the subset of S all containing the same value of the attribute
        gain = S_metric - expected_metric_sum
        # Round to zero to prevent floating point errors compounding
        if (math.isclose(gain, 0.00, abs_tol=1e-7)):
            gain = 0.00
        return (gain, Sv_dict)

    @classmethod
    def get_attribute_values(cls, S, attribute):
        # Should likely just use get_value_subset and look at keys but this may be faster if subset isn't needed
        attribute_vals = set()
        for example in S:
            ex_val = example[attribute]
            attribute_vals.add(ex_val)
        return attribute_vals

    @classmethod
    def get_value_subset(cls, S, labels, attribute, weights):
        if (len(S) != len(labels)):
            raise Exception('S must be the same length as corresponding labels')
        if (len(weights) != len(labels)):
            raise Exception('Weights must be the same length as corresponding labels')
        Sv_dict = {}
        for i in range(len(S)):
            example = S[i]
            label = labels[i]
            weight = weights[i]
            ex_val = example[attribute]
            if ex_val in Sv_dict:
                Sv_dict[ex_val][0].append(example)
                Sv_dict[ex_val][1].append(label)
                Sv_dict[ex_val][2].append(weight)
            else:
                Sv_dict[ex_val] = ([example],[label],[weight])
        # a dictionary that given a value of the attribute maps to a tuple of (Sv, labels_v) representing the subset of S all containing the same value of the attribute
        return Sv_dict

    @classmethod
    def find_best_attribute(cls, S, labels, attributes, weights, metric=None):
        best_Sv_dict = None 
        best_attribute = None
        best_gain = 0
        for current_attribute in attributes:
            (current_gain, current_Sv_dict) = cls.gain(S, labels, current_attribute, weights, metric=metric)
            # Will choose best gain based on this equality, if multiple gains of same value will arbitrarily choose one
            if(current_gain >= best_gain):
                best_Sv_dict = current_Sv_dict
                best_attribute = current_attribute
                best_gain = current_gain
        # Returns a tuple of the best attribute, a dictionary maps to a tuple of (Sv, labels_v) representing the subsets of S for splitting on the best attribute, and the best gain value
        return (best_attribute, best_Sv_dict, best_gain)

    @classmethod
    def prediction_error(cls, y_pred, y_actual):
        # Determine prediction error 
        if len(y_pred) != len(y_actual):
            raise Exception('y_pred and y_actual are not the same length')
        error_count = 0
        for i in range(len(y_pred)):
            if y_pred[i] != y_actual[i]:
                error_count +=1
        return error_count/len(y_pred)

    @classmethod
    def extract_ID3_input(cls, filename, attributes, attribute_discretize_idx_to_thresh_map={}, unknown_value='unknown', unknown_replacement_map={}):
        """ Extracts ID3 input given a csv file

        Args:
            filename (str): File to extract ID3 input from 
            attributes (list[str]): Names of attributes that sequentially correspond to columns in the data file 
            attribute_discretize_idx_to_thresh_map (dict, optional): Dictionary that specifies which attributes to discretize by index (AKA zero-based column numbers) as keys with the value being the treshold value to perform the discretization. Defaults to {}.
            unknown_value (str, optional): Name of unknown value in data to search for when replacing with different value. Defaults to 'unknown'.
            unknown_replacement_map (dict, optional): A dictionary that maps from an attribute to the desired replacement value for a given 'unknown', if empty or can't find mapping for attribute of unknown value will perform no mapping. Defaults to {}.

        Returns:
            tuple: (S, attributes, labels, attribute_possible_vals_in_this_data) for usage with ID3. 
        """
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
            attribute_possible_vals_in_this_data[a] = list(cls.get_attribute_values(S, a))
        return (S, attributes, labels, attribute_possible_vals_in_this_data)

    @classmethod
    def get_attribute_discretize_idx_to_thresh_map(cls, filename, attributes, attribute_idx_to_discretize):
    # Gets dictionary that specifies which attributes to discretize by index (AKA zero-based column numbers) as keys with the value being the treshold value to perform the discretization.
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

    @classmethod
    def get_attribute_to_most_common_value_map(cls, filename, attributes):
    # Gets a dictionary that maps from an attribute to the desired replacement value for a given 'unknown'.
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