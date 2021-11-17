# Libraries
import numpy as np

class Svm:
    def __init__(self):
        """ Create least Support Vector Machine model that can be trained using linear primal form stochastic sub-gradient descent, or using the kernalized dual form and quadratic optimization.
        """

        self.w = np.array([])


    def train_primal_stochastic_subgradient_descent(self, X, y, T=1000, abs_obj_val_diff_thresh=1e-6, r=0.01, C=0.1, r_sched=None):
        """ Train using using primal objective based stochastic sub-gradient descent with randomized sampling. Will set the weight vector property of this object to the value found from training.

        Args:
            X (ndarray): An m by d 2D numpy array where m is the number of examples and d is the number of features. 
            y (ndarray): A m 1D numpy array where m is the number of examples.
            T (int, optional): Maximum number of epochs to train for. Defaults to 1000. 
            abs_obj_val_diff_thresh (float, optional): The threshold to stop stochastic gradient descent (what is considered convergence), the absolute value between the current and previous objective function values. Defaults to 1e-6.
            r (float, optional): Learning rate for gradient descent. Defaults to 0.01
            C (float, optional): Tradeoff hyperparameter between empirical loss (hinge loss, penalizing mistakes) and regularization term (maximize margin term). Higher C means higher weight to empirical loss term. Defaults to 0.1.
            r_sched (ndarray, optional): Learning rate schedule for each epoch (will override usage of r since r for each epoch was given). Defaults to None which means not learning rate schedule is used and constant r is used for each epoch.

        Returns:
            list: Returns a list containing the objective function value at each epoch. 
        """

        objective_vals = []
        num_examples, num_features = X.shape
        if r_sched is not None and len(r_sched) != T:
            raise Exception('Not enough scheduled learning rates for number of maximum epochs')

        # Augment X and w
        num_features_augmented = num_features + 1
        X_aug = np.append(X, np.ones((X.shape[0],1)), axis=1)
        self.w = np.zeros(num_features_augmented)

        example_idxs_shuffle = np.arange(num_examples)
        current_objective_val = 0
        for t in range(T):
            np.random.shuffle(example_idxs_shuffle) # Shuffle in-place
            r_t = r if r_sched is None else r_sched[t]
            for i in example_idxs_shuffle:
                # Update weights
                if (y[i] * np.dot(self.w, X_aug[i,:])) <= 1:
                    w_zero_augmented = self.w.copy()
                    w_zero_augmented[-1] = 0.0
                    self.w = self.w - r_t*w_zero_augmented + r_t*C*num_examples*y[i]*X_aug[i,:]
                else:
                    self.w[:-1] = (1-r_t) * self.w[:-1]

            # Calculate and update current cost
            new_objective_val = Svm.primal_objective_function_value(X_aug, y, self.w, C)
            abs_cost_diff = abs(new_objective_val - current_objective_val)
            current_objective_val = new_objective_val
            objective_vals.append(current_objective_val)

            # Break if the convergence critera is met
            if abs_cost_diff < abs_obj_val_diff_thresh:
                break
        return objective_vals
    

    def primal_predict(self, X):
        """ Predict using primal SVM objective 

        Args:
            X (ndarray): An m by d 2D numpy array where m is the number of examples and d is the number of features.

        Returns:
            int: Predicted class
        """
        X_aug = np.append(X, np.ones((X.shape[0],1)), axis=1)
        return np.sign(np.dot(X_aug, self.w))

    @classmethod
    def primal_objective_function_value(cls, X, y, w, C):
        """ Calculates the SVM primal objective function value

        Args:
            X (ndarray): An m by d+1 2D numpy array where m is the number of examples and d is the number of features and where the last column is a column of ones as this is an augmented X.
            y (ndarray): A m 1D numpy array where m is the number of examples.
            w (ndarray): Augmented weight vector. A d+1 1D numpy array where d is the number of features.
            C (float): Tradeoff hyperparameter between empirical loss (hinge loss, penalizing mistakes) and regularization term (maximize margin term). Higher C means higher weight to empirical loss term.

        Returns:
            float: Objective function value 
        """
        J = 0.5 * np.dot(w[:-1], w[:-1]) + C * np.sum(np.maximum(0, (1 - y * np.dot(X, w))))
        return J

        
