import collections
import numpy as np
from numpy.core.fromnumeric import clip

def calculate_prediction_error(y_pred, y):
    """ Get prediction error.

    Args:
        y_pred (ndarray): Predicted label data. A m shape 1D numpy array where m is the number of examples.
        y (ndarray): Ground truth label data. A m shape 1D numpy array where m is the number of examples.

    Returns:
        float: Prediction error.
    """
    return np.sum(y_pred!=y) / y.shape[0]

def calculate_correct_examples(y_pred, y):
    """ Get count of correctly predicted labels.

    Args:
        y_pred (ndarray): Binary (1 or -1) predicted label data. A m shape 1D numpy array where m is the number of examples.
        y (ndarray): Binary (1 or -1) ground truth label data. A m shape 1D numpy array where m is the number of examples. 

    Returns:
        int: Count of correctly predicted labels.
    """
    return np.sum(y_pred==y)

class Perceptron():
    def __init__(self):
        """ Create a trainable standard perceptron.
        """
        self.w = np.array([])

    def train(self, X, y, T=10, r=0.01):
        """ Train a standard perceptron.

        Args:
            X (ndarray): Feature data. An m by d shape 2D numpy array where m is the number of examples and d is the number of features. 
            y (ndarray): Binary (1 or -1) label data. A m shape 1D numpy array where m is the number of examples.
            T (int, optional): Number of epochs to run perceptron over the entire set of examples. Defaults to 10.
            r (float, optional): Learning rate. Defaults to 0.01.

        Returns:
            ndarray: A d shape 1D numpy array representing the learned weight vector from training where d is the number of features. 
        """
        num_examples, num_features = X.shape
        self.w = np.zeros(num_features)
        for _ in range(T):
            for i in range(num_examples):
                xi = X[i,:]
                yi = y[i]
                if (yi*np.dot(self.w, xi) <= 0):
                    self.w = self.w + r * (yi*xi)
        return self.w

    def predict(self, X):
        """ Predict using a standard perceptron.

        Args:
            X (ndarray): Feature data. An m by d shape 2D numpy array where m is the number of examples and d is the number of features. 

        Returns:
            ndarray: A n shape 1D numpy array representing the predictions of the perceptron with the current learned weight vector. 
        """
        return np.sign(np.dot(X, self.w))



class VotedPerceptron():
    def __init__(self):
        """ Create a trainable voted perceptron.
        """
        self.w = np.array([[]]) # A k by d shape 2D numpy array of all k learned weight vectors
        self.c = np.array([]) # A k shape 1D numpy array of the vote for all k learned weight vectors

    def train(self, X, y, T=10, r=0.01):
        """ Train a voted perceptron.

        Args:
            X (ndarray): Feature data. An m by d shape 2D numpy array where m is the number of examples and d is the number of features. 
            y (ndarray): Binary (1 or -1) label data. A m shape 1D numpy array where m is the number of examples.
            T (int, optional): Number of epochs to run perceptron over the entire set of examples. Defaults to 10.
            r (float, optional): Learning rate. Defaults to 0.01.

        Returns:
            tuple: tuple containing 
                ndarray: A k by d shape 2D numpy array of all k learned weight vectors
                ndarray: A k shape 1D numpy array of the vote for all k learned weight vectors
        """
        num_examples, num_features = X.shape
        w_list = []
        c_list = []
        w = np.zeros(num_features)
        c = 0 
        for _ in range(T):
            for i in range(num_examples):
                xi = X[i,:]
                yi = y[i]
                if (yi*np.dot(w, xi) <= 0):
                    if (i != 0):
                        w_list.append(w.copy()) # Append a new wm
                        c_list.append(c) # Appends a new cm
                    w = w + r * (yi*xi)
                    c = 1
                else:
                    c += 1
        # Now have k items in w_list and c_list
        self.w = np.array(w_list) # Shape kxd
        self.c = np.array(c_list) # Shape k
        return (self.w, self.c)

    def predict(self, X):
        """ Predict using a voted perceptron.

        Args:
            X (ndarray): Feature data. An m by d shape 2D numpy array where m is the number of examples and d is the number of features. 

        Returns:
            ndarray: A n shape 1D numpy array representing the predictions of the perceptron with the current learned weight vector. 
        """
        num_examples, num_features = X.shape
        pred_sum = np.zeros_like(num_examples)
        for i in range(self.c.shape[0]):
            current_pred = np.sign(np.dot(X, self.w[i,:]))
            pred_sum = pred_sum + self.c[i] * current_pred
        return np.sign(pred_sum)



class AveragedPerceptron():
    def __init__(self):
        """ Create a trainable averaged perceptron.
        """
        self.w = np.array([]) 
        self.a = np.array([]) 

    def train(self, X, y, T=10, r=0.01):
        """ Train an averaged perceptron.

        Args:
            X (ndarray): Feature data. An m by d shape 2D numpy array where m is the number of examples and d is the number of features. 
            y (ndarray): Binary (1 or -1) label data. A m shape 1D numpy array where m is the number of examples.
            T (int, optional): Number of epochs to run perceptron over the entire set of examples. Defaults to 10.
            r (float, optional): Learning rate. Defaults to 0.01.

        Returns:
            tuple: tuple containing 
                ndarray: Final weight vector A d shape 1D numpy array. 
                ndarray: ndarray: Sum of all the weight vectors that were explored. A d shape 1D numpy array.
            
        """
        num_examples, num_features = X.shape
        self.w = np.zeros(num_features)
        self.a = np.zeros(num_features)
        for _ in range(T):
            for i in range(num_examples):
                xi = X[i,:]
                yi = y[i]
                if (yi*np.dot(self.w, xi) <= 0):
                    self.w = self.w + r * (yi*xi)
                self.a = self.a + self.w
        return (self.w, self.a)

    def predict(self, X):
        """ Predict using an averaged perceptron.

        Args:
            X (ndarray): Feature data. An m by d shape 2D numpy array where m is the number of examples and d is the number of features. 

        Returns:
            ndarray: A n shape 1D numpy array representing the predictions of the perceptron with the current learned weight vector. 
        """
        return np.sign(np.dot(X, self.a))