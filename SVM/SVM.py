# Libraries
import numpy as np
from scipy import optimize
from scipy.optimize import minimize, Bounds

class Svm:
    def __init__(self):
        """ Create least Support Vector Machine model that can be trained using linear primal form stochastic sub-gradient descent, or using the kernalized dual form and quadratic optimization.
        Once trained can access optimal augmented weight vector containing optimal bias as the last element.
        """

        self.w_aug = np.array([])

    def train_dual(self, X, y, C=0.1):
        # Optimize dual objective function
        # alpha_0 = np.full(y.shape, C/2)
        alpha_0 = np.zeros(y.shape)
        sum_alpha_y_constraint = {'type':'eq', 'fun':self._sum_alpha_y_constraint, 'args':(y,)}
        alpha_bound = Bounds(0, C)
        dual_obj_result = minimize(self._dual_objective_function_value, alpha_0, args=(X, y), method='SLSQP', bounds=alpha_bound, constraints=sum_alpha_y_constraint, options={'disp':True})
        alpha_star = dual_obj_result.x

        # Form notions of optimal w and b
        w_star = np.dot(alpha_star*y,X)  # Non augmented optimal w
        b_star = np.array([np.mean(y - np.dot(X, w_star))])
        self.w_aug = np.concatenate((w_star, b_star))

        return alpha_star
    
    def _dual_objective_function_value(self, alpha, X, y):
        # Non-vectorized
        # N = y.shape[0]
        # y_alpha_x_sum = 0
        # for i in range(N):
        #     for j in range(N):
        #         y_alpha_x_sum += 0.5 * y[i] * y[j] * alpha[i] * alpha[j] * np.dot(X[i,:], X[j,:])
        # print(y_alpha_x_sum)

        # Vectorized, specifically using einstein summing notation (indices are like "dummy" variables, shared index in multiply implies a sum, free subscript indicates and index)
        y_alpha_x_sum = 0.5 * np.einsum('i,j,i,j,ix,jx->', y, y, alpha, alpha, X, X, optimize='optimal')
        # print(y_alpha_x_sum)
        # print()
       
        alpha_sum = np.sum(alpha)
        return y_alpha_x_sum - alpha_sum

    def _sum_alpha_y_constraint(self, alpha, y):
        return np.dot(alpha, y)


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
        self.w_aug = np.zeros(num_features_augmented)

        example_idxs_shuffle = np.arange(num_examples)
        current_objective_val = 0
        for t in range(T):
            np.random.shuffle(example_idxs_shuffle) # Shuffle in-place
            r_t = r if r_sched is None else r_sched[t]
            for i in example_idxs_shuffle:
                # Update weights
                if (y[i] * np.dot(self.w_aug, X_aug[i,:])) <= 1:
                    w_zero_augmented = self.w_aug.copy()
                    w_zero_augmented[-1] = 0.0
                    self.w_aug = self.w_aug - r_t*w_zero_augmented + r_t*C*num_examples*y[i]*X_aug[i,:]
                else:
                    self.w_aug[:-1] = (1-r_t) * self.w_aug[:-1]

            # Calculate and update current cost
            new_objective_val = self._primal_objective_function_value(X_aug, y, self.w_aug, C)
            abs_cost_diff = abs(new_objective_val - current_objective_val)
            current_objective_val = new_objective_val
            objective_vals.append(current_objective_val)

            # Break if the convergence critera is met
            if abs_cost_diff < abs_obj_val_diff_thresh:
                break
        return objective_vals
    
    def _primal_objective_function_value(self, X_aug, y, w_aug, C):
        """ Calculates the SVM primal objective function value

        Args:
            X_aug (ndarray): An m by d+1 2D numpy array where m is the number of examples and d is the number of features and where the last column is a column of ones as this is an augmented X.
            y (ndarray): A m 1D numpy array where m is the number of examples.
            w_aug (ndarray): Augmented weight vector. A d+1 1D numpy array where d is the number of features.
            C (float): Tradeoff hyperparameter between empirical loss (hinge loss, penalizing mistakes) and regularization term (maximize margin term). Higher C means higher weight to empirical loss term.

        Returns:
            float: Objective function value 
        """
        J = 0.5 * np.dot(w_aug[:-1], w_aug[:-1]) + C * np.sum(np.maximum(0, (1 - y * np.dot(X_aug, w_aug))))
        return J

    def predict(self, X):
        """ Predict using trained SVM.

        Args:
            X (ndarray): An m by d 2D numpy array where m is the number of examples and d is the number of features.

        Returns:
            int: Predicted class
        """
        X_aug = np.append(X, np.ones((X.shape[0],1)), axis=1)
        return np.sign(np.dot(X_aug, self.w_aug))


        
