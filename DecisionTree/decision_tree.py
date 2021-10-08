import math

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

    def train(self, S, attributes, labels, metric=None, max_depth=None):
        """ Trains decision tree on labelled data

        Args:
            S (list of dict[str: str]): A list with each element being a dictionary which represents an example. This dictionary maps from each possible attribute to the value of the attribute for the example. 
            attributes (list[str]): A list containing each possible attribute value 
            labels (list): A list of the labels for each example in S. Must be same order and length as S. 
            metric (Callable[[list], float], optional): Callable function representing the metric (heuristic) to use to determine how to split decision tree when training. Defaults to None which will use entropy as the heuristic.
            max_depth (int, optional): Maximum depth of decision tree, if examples not perfectly split and max_depth is hit will choose most common class as leaf node. Defaults to None which means tree depth will not be limited.

        Returns:
            Node: Root node of the Decision tree. 
        """
        self.root = self.ID3(S, attributes, labels, metric=metric, max_depth=max_depth)
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
        
    def ID3(self, S, attributes, labels, metric=None, max_depth=None):
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
                root_node.add_child(self.ID3(Sv, reduced_attributes, labels_v, metric=metric, max_depth=max_depth), value)
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
            if(current_gain >= best_gain):
                best_Sv_dict = current_Sv_dict
                best_attribute = current_attribute
                best_gain = current_gain
        # Returns a tuple of the best attribute, a dictionary maps to a tuple of (Sv, labels_v) representing the subsets of S for splitting on the best attribute, and the best gain value
        return (best_attribute, best_Sv_dict, best_gain)
