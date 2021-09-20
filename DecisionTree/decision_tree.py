import math
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

class DecisionTree:
    def __init__(self):
        self.root = Node('root')
        # self.current_node = self.root

    @staticmethod
    def entropy(labels):
        N = len(labels)
        possible_labels = set(labels)
        entropy = 0
        for label in possible_labels:
            label_count = labels.count(label)
            label_prob = label_count/N
            if(label_prob != 0):
                entropy += -label_prob*math.log(label_prob)
        return entropy

    @staticmethod
    def majority_error(labels):
        N = len(labels)
        possible_labels = set(labels)
        most_common_label = max(possible_labels, key=labels.count)
        non_most_common_count = 0
        non_most_common_labels = possible_labels - {most_common_label}
        for label in non_most_common_labels:
            non_most_common_count += labels.count(label)
        return non_most_common_count/N

    @staticmethod
    def gini_index(labels):
        N = len(labels)
        possible_labels = set(labels)
        label_prob_sum = 0
        for label in possible_labels:
            label_count = labels.count(label)
            label_prob = label_count/N
            label_prob_sum += label_prob**2 
        return 1-label_prob_sum

    @staticmethod
    def gain(S, labels, attribute, metric=entropy):
        N = len(S)
        S_metric = metric(labels)
        expected_metric_sum = 0
        Sv_dict = DecisionTree.get_value_subset(S, labels, attribute)
        for Sv_key in Sv_dict.keys():
            Sv = Sv_dict[Sv_key]
            labels_v = [i[1] for i in Sv]
            expected_metric_sum += (len(Sv)/N) * metric(labels_v)
        # Returns gain value and the splitted subsets from splitting on attribute
        return (S_metric - expected_metric_sum, Sv_dict)

    @staticmethod
    def get_attribute_values(S, attribute):
        attribute_vals = set()
        for example in S:
            ex_val = example[attribute]
            attribute_vals.add(ex_val)
        return attribute_vals

    @staticmethod
    def get_value_subset(S, labels, attribute):
        if (len(S) != len(labels)):
            raise Exception('S must be the same length as corresponding labels')
        Sv_dict = {}
        for i in range(len(S)):
            example = S[i]
            label = labels[i]
            ex_val = example[attribute]
            if ex_val in Sv_dict:
                Sv_dict[ex_val].append((example, label))
            else:
                Sv_dict[ex_val] = [(example, label)]
        return Sv_dict



    def ID3_train(self, S, attributes, labels, max_depth=None):
        pass

        # if len(labels) <= 0:
        #     raise Exception('Must have at least a single example or label or attribute')

        # # All examples have same label
        # if(labels.count(labels[0]) == len(labels)):
        #     return
        # # Attributes empty
        # if(len(attributes) <= 0):
        #     return

        # current_node = Node('root')

        # best_attr = self.find_best_attr(S, 'entropy')

        # current_node.name = best_attr

        # best_attr_vals = set()
        # for example in S:
        #     best_attr_vals.add(example[best_attr])
        # for val in best_attr_vals:
        #     branch = Node(val)
        #     current_node.add_child(Node())

        





# class DecisionTree:
#     _root

    
