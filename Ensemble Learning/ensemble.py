import sys
sys.path.insert(1, '../')
from DecisionTree.decision_tree import DecisionTree
import numpy as np

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
        """ Creates AdaBoosted decision stump based model object that to train and predict. Takes a description of possible attributes and their possible values that can be trained and then used for prediction.

        Args:
            attribute_possible_vals (dict of str: list[str]): A dictionary mapping from all possible attributes to a list containing every possible value for the attribute
        """
        self.stump_alpha_t_list = []
        self.attribute_possible_vals = attribute_possible_vals

    def train(self, S, attributes, labels, T):
        """ Trains an AdaBoosted decision stump model

        Args:
            S (list of dict[str: str]): A list with each element being a dictionary which represents an example. This dictionary maps from each possible attribute to the value of the attribute for the example. 
            attributes (list[str]): A list containing each possible attribute value 
            labels (list): A list of the labels for each example in S. Must be same order and length as S. 
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


class BaggedTrees:
    def __init__(self) -> None:
        pass

class RandomForest:
    def __init__(self) -> None:
        pass