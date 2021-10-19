# Libraries
import sys
import numpy as np
import random

# Personal Libraries
sys.path.insert(1, '../')
from DecisionTree.decision_tree import DecisionTree

class DecisionStump:
    def __init__(self, attribute_possible_vals):
        """ Creates DecisionTree object with only two levels (a depth of 1) with a description of possible attributes and their possible values that can be trained and then used for prediction.

        Args:
            attribute_possible_vals (dict of str: list[str]): A dictionary mapping from all possible attributes to a list containing every possible value for the attribute
        """
        self.attribute_possible_vals = attribute_possible_vals
        self.root = None
        self.tree = DecisionTree(attribute_possible_vals)
    def train(self, S, attributes, labels, S_weights, metric=None):
        """ Trains decision stump on labelled data

        Args:
            S (list of dict[str: str]): A list with each element being a dictionary which represents an example. This dictionary maps from each possible attribute to the value of the attribute for the example. 
            attributes (list[str]): A list containing each possible attribute value 
            labels (list): A list of the labels for each example in S. Must be same order and length as S. 
            S_weights (list[float]): Weights (probabilities) associated with each training example. Must be same length as labels and values should sum to 1.
            metric (Callable[[list], float], optional): Callable function representing the metric (heuristic) to use to determine how to split decision stump when training. Defaults to None which will use entropy as the heuristic.

        Returns:
            Node: Root node of the Decision stump. 
        """
        self.root = self.tree.train(S, attributes, labels, metric=metric, max_depth=1, weights=S_weights)
        return self.root
    def predict(self, S):
        """ Predicts on already trained decision stump.

        Args:
            S (list of dict): A list with each element being a dictionary which represents an example. This dictionary maps from each possible attribute to the value of the attribute for the example. 

        Returns:
            List of predicted labels corresponding to each example given in S.
        """
        return self.tree.predict(S)

class AdaBoost:
    def __init__(self, attribute_possible_vals):
        """ Creates AdaBoosted binary decision stump based model object that to train and predict. Takes a description of possible attributes and their possible values that can be trained and then used for prediction.

        Args:
            attribute_possible_vals (dict of str: list[str]): A dictionary mapping from all possible attributes to a list containing every possible value for the attribute
        """
        self.stump_alpha_t_list = []
        self.attribute_possible_vals = attribute_possible_vals

    def train(self, S, attributes, labels, T):
        """ Trains an AdaBoosted binary decision stump model

        Args:
            S (list of dict[str: str]): A list with each element being a dictionary which represents an example. This dictionary maps from each possible attribute to the value of the attribute for the example. 
            attributes (list[str]): A list containing each possible attribute value 
            labels (list): A list of the labels for each example in S. Must be same order and length as S. Labels must be binary, either -1 or +1.
            S_weights (list[float]): Weights (probabilities) associated with each training example. Must be same length as labels and values should sum to 1.
            T (int) the number of epochs to run AdaBoost.
        """
        self.stump_alpha_t_list = []
        y = np.array(labels)
        m = len(S)
        D = np.full(m,1/m)
        for t in range(T):
            # Get a by creating stump then training it to create a hypothesis
            stump = DecisionStump(self.attribute_possible_vals)
            stump.train(S, attributes, labels, D.tolist())
            hS = np.array(stump.predict(S))
            epsilon_t = 0.5 - 0.5 * np.sum(D * y * hS)
            if (epsilon_t >= 0.5):
                raise Exception('Hypothesis weighted error was not less than chance, weighted error was: {0}'.format(epsilon_t))
            alpha_t = 0.5 * np.log((1-epsilon_t)/epsilon_t)
            # Store this to use for prediction 
            self.stump_alpha_t_list.append((stump, alpha_t))

            # Update D
            D = D * np.exp(-alpha_t * y * hS)
            norm_const = 1/np.sum(D)
            D =  norm_const * D # Make sure D sums to 1

    def predict(self, S):
        """ Predicts on an already trained AdaBoosted decision stump model.

        Args:
            S (list of dict): A list with each element being a dictionary which represents an example. This dictionary maps from each possible attribute to the value of the attribute for the example. 

        Returns:
            List of predicted labels corresponding to each example given in S.
        """
        alpha_t_hS_sum = np.zeros_like(S)
        for stump, alpha_t in self.stump_alpha_t_list:
            alpha_t_hS_sum += (alpha_t * np.array(stump.predict(S)))
        return np.sign(alpha_t_hS_sum).tolist()


class BaggedDecisionTree:
    def __init__(self, attribute_possible_vals):
        """ Builds a binary bagged decision tree object. 

        Args:
            attribute_possible_vals (dict of str: list[str]): A dictionary mapping from all possible attributes to a list containing every possible value for the attribute
        """
        self.decision_tree_list = []
        self.attribute_possible_vals = attribute_possible_vals

    def train(self, S, attributes, labels, T):
        """ Trains a binary bagged decision tree with T decision trees built from out-of-bag samples.

        Args:
            S (list of dict[str: str]): A list with each element being a dictionary which represents an example. This dictionary maps from each possible attribute to the value of the attribute for the example. 
            attributes (list[str]): A list containing each possible attribute value 
            labels (list): A list of the labels for each example in S. Must be same order and length as S. Labels must be binary, either -1 or +1.
            S_weights (list[float]): Weights (probabilities) associated with each training example. Must be same length as labels and values should sum to 1.
            T (int) the number of decision trees to build from out-of-bag samples.
        """
        self.decision_tree_list = []
        for t in range(T):
            self.train_one_extra_tree(S, attributes, labels)

    def train_one_extra_tree(self, S, attributes, labels):
        """ Trains one extra decision tree built from out-of-bag samples. Can be used to test what happens when T is grown without having to retrain an extensive number of trees repetitively.

        Args:
            S (list of dict[str: str]): A list with each element being a dictionary which represents an example. This dictionary maps from each possible attribute to the value of the attribute for the example. 
            attributes (list[str]): A list containing each possible attribute value 
            labels (list): A list of the labels for each example in S. Must be same order and length as S. Labels must be binary, either -1 or +1.
            S_weights (list[float]): Weights (probabilities) associated with each training example. Must be same length as labels and values should sum to 1.
        """
        S_sampled, labels_sampled = BaggedDecisionTree.uniform_sample_with_replacement(S, labels)
        tree = DecisionTree(self.attribute_possible_vals)
        tree.train(S_sampled, attributes, labels_sampled)
        self.decision_tree_list.append(tree)

    @classmethod
    def uniform_sample_with_replacement(cls, S, labels, sample_size=None):
        S_sampled = []
        labels_sampled = []
        if sample_size is None:
            m = len(S)
        else:
            m = sample_size
        zipped_data = list(zip(S, labels))
        for i in range(m):
            s,l = random.choice(zipped_data)
            S_sampled.append(s)
            labels_sampled.append(l)
        return S_sampled, labels_sampled

    def predict(self, S):
        """ Predicts using all decision trees built from out-of-bag samples by taking votes from each. If equal number of votes for each binary class will select the positive(+1) class.

        Args:
            S (list of dict): A list with each element being a dictionary which represents an example. This dictionary maps from each possible attribute to the value of the attribute for the example. 

        Returns:
            List of predicted labels corresponding to each example given in S.
        """
        pred_sum = np.zeros_like(S)
        for tree in self.decision_tree_list:
            pred_sum += np.array(tree.predict(S))
        result = np.sign(pred_sum)
        # Replace all with equal votes (sign returns 0) with positive class
        result[result == 0] = 1
        return result.tolist()


class RandomForest:
    def __init__(self, attribute_possible_vals):
        """ Build random forest object. 

        Args:
            attribute_possible_vals (dict of str: list[str]): A dictionary mapping from all possible attributes to a list containing every possible value for the attribute
        """
        self.decision_tree_list = []
        self.attribute_possible_vals = attribute_possible_vals

    def train(self, S, attributes, labels, T, random_forest_sampling_size=None):
        """ Trains a random forest with T decision trees built from out-of-bag samples and looking at random small subsets of the attributes as candidates to split on.

        Args:
            S (list of dict[str: str]): A list with each element being a dictionary which represents an example. This dictionary maps from each possible attribute to the value of the attribute for the example. 
            attributes (list[str]): A list containing each possible attribute value 
            labels (list): A list of the labels for each example in S. Must be same order and length as S. Labels must be binary, either -1 or +1.
            S_weights (list[float]): Weights (probabilities) associated with each training example. Must be same length as labels and values should sum to 1.
            T (int) the number of decision trees to build from out-of-bag samples.
            random_forest_sampling_size (int, optional): Will try to use sample this subset size of attributes (if less attributes than number gives will use all attributes). Defaults to None which means the ceiling of the current attribute size / 5 is used.
        """
        self.decision_tree_list = []
        for t in range(T):
            self.train_one_extra_tree(S, attributes, labels, random_forest_sampling_size=random_forest_sampling_size)

    def train_one_extra_tree(self, S, attributes, labels, random_forest_sampling_size=None):
        """ Trains one extra decision tree built from out-of-bag samples and looking at random small subsets of the attributes as candidates to split on. Can be used to test what happens when T is grown without having to retrain an extensive number of trees repetitively.

        Args:
            S (list of dict[str: str]): A list with each element being a dictionary which represents an example. This dictionary maps from each possible attribute to the value of the attribute for the example. 
            attributes (list[str]): A list containing each possible attribute value 
            labels (list): A list of the labels for each example in S. Must be same order and length as S. Labels must be binary, either -1 or +1.
            S_weights (list[float]): Weights (probabilities) associated with each training example. Must be same length as labels and values should sum to 1.
            random_forest_sampling_size (int, optional): Will try to use sample this subset size of attributes (if less attributes than number gives will use all attributes). Defaults to None which means the ceiling of the current attribute size / 5 is used.
        """
        S_sampled, labels_sampled = RandomForest.uniform_sample_with_replacement(S, labels)
        tree = DecisionTree(self.attribute_possible_vals) 
        tree.train(S_sampled, attributes, labels_sampled, random_forest_attribute_sampling=True, random_forest_sampling_size=random_forest_sampling_size)
        self.decision_tree_list.append(tree)

    @classmethod
    def uniform_sample_with_replacement(cls, S, labels, sample_size=None):
        S_sampled = []
        labels_sampled = []
        if sample_size is None:
            m = len(S)
        else:
            m = sample_size
        zipped_data = list(zip(S, labels))
        for i in range(m):
            s,l = random.choice(zipped_data)
            S_sampled.append(s)
            labels_sampled.append(l)
        return S_sampled, labels_sampled

    def predict(self, S):
        """ Predicts using all decision trees built from out-of-bag samples and looking at random small subsets of the attributes as candidates to split on by taking votes from each. If equal number of votes for each binary class will select the positive(+1) class.

        Args:
            S (list of dict): A list with each element being a dictionary which represents an example. This dictionary maps from each possible attribute to the value of the attribute for the example. 

        Returns:
            List of predicted labels corresponding to each example given in S.
        """
        pred_sum = np.zeros_like(S)
        for tree in self.decision_tree_list:
            pred_sum += np.array(tree.predict(S))
        result = np.sign(pred_sum)
        # Replace all with equal votes (sign returns 0) with positive class
        result[result == 0] = 1
        return result.tolist()