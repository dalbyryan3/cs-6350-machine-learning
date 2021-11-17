# Libraries
from math import gamma
import numpy as np
from scipy.optimize import minimize, Bounds

class Svm:
    def __init__(self):
        """ Create least Support Vector Machine model that can be trained using linear primal form stochastic sub-gradient descent, or using the kernalized dual form and quadratic optimization.
        If trained using primal or non-kernelized dual can access optimal augmented weight vector containing optimal bias as the last element which are all necessary for prediction.
        If trained using rbf-kernelized dual can access training labels, featrues, bias, and optimal alpha values which all necessary for prediction. 
        """
        self._set_to_default_parameter_values()

    def _set_to_default_parameter_values(self, rbf_kernel=False):
        self.rbf_kernel = rbf_kernel
        self.w_aug = np.array([])
        self.rbf_kernel_pred_data = {'y':None, 'X':None, 'alpha_star':None, 'b':None} 

    def train_dual(self, X, y, C=0.1, rbf_kernel=False, rbf_gamma=0.1):
        # Set to default parameter values
        self._set_to_default_parameter_values(rbf_kernel)

        obj_func = self._rbf_kernel_dual_objective_function_value if self.rbf_kernel else self._dual_objective_function_value
        obj_func_extra_args = (X, y, rbf_gamma) if self.rbf_kernel else (X, y)
        # Optimize dual objective function
        alpha_0 = np.zeros(y.shape)
        sum_alpha_y_constraint = {'type':'eq', 'fun':self._sum_alpha_y_constraint, 'args':(y,)}
        alpha_bound = Bounds(0, C)
        dual_obj_result = minimize(obj_func, alpha_0, args=obj_func_extra_args, method='SLSQP', bounds=alpha_bound, constraints=sum_alpha_y_constraint, options={'disp':True})
        alpha_star = dual_obj_result.x

        # TODO HANDLE PREDICTION USING RBF (should save alpha, y, X, and b(need to calculate using kernel formulation- could save as dict or sequence obj called high_dim_lifting_map_data)) - maybe have flag if last training was using rbf_kernel? -will have to update homework 4 to output correct information 
        # Should also note in documentation that if rbf_kernel flag is true then high_dim_lifting_map_data is set otherwise then w_aug is set instead- SHOULD CLEAR THESE AT BEGINING OF train_dual
        # Will return alpha star since technically it is the main output of the optimizer and rest can be found form it

        if (self.rbf_kernel):
            self.rbf_kernel_pred_data['y'] = y
            self.rbf_kernel_pred_data['X'] = X
            self.rbf_kernel_pred_data['alpha_star'] = alpha_star
            # TODO calculate b
        else:
            # Form notions of optimal w and b
            w_star = np.dot(alpha_star*y,X)  # Non augmented optimal w
            b_star = np.array([np.mean(y - np.dot(X, w_star))])
            self.w_aug = np.concatenate((w_star, b_star))

        return alpha_star
    
    def _rbf_kernel_dual_objective_function_value(self, alpha, X, y, rbf_gamma):
        # Non-vectorized
        # m = y.shape[0]
        # y_alpha_kernel_sum = 0
        # rbf_kernel_vec_list = []
        # for i in range(m):
        #     rbf_kernel_list = []
        #     for j in range(m):
        #         rbf_kernel_val = np.exp(-(np.linalg.norm(X[i,:] - X[j,:])**2)/rbf_gamma)
        #         rbf_kernel_list.append(rbf_kernel_val)
        #         y_alpha_kernel_sum += 0.5 * y[i] * y[j] * alpha[i] * alpha[j] * rbf_kernel_val
        #     rbf_kernel_vec_list.append(rbf_kernel_list)
        # rbf_kernel_mat = np.array(rbf_kernel_vec_list)
        # print(np.sum(rbf_kernel_mat))
        # print("shape kern = {0}".format(rbf_kernel_mat.shape))
        # print(y_alpha_kernel_sum)

        # Vectorized, specifically using einstein summing notation (indices are like "dummy" variables, shared index in multiply implies a sum, free subscript indicates and index)
        # Also squared pairwise distance was formed as a vector operation
        X_norm = np.einsum('ij,ij->i', X, X)
        rbf_pairwise_dist_sq = X_norm[:,np.newaxis] + X_norm[np.newaxis,:] - 2 * np.dot(X, X.T) # size mxm, corresponding pairwise distance for any i,j
        rbf_kernel_mat = np.exp(-rbf_pairwise_dist_sq/rbf_gamma) 
        y_alpha_kernel_sum = 0.5 * np.einsum('i,j,i,j,ij->', y, y, alpha, alpha, rbf_kernel_mat, optimize='optimal')
        # print(np.sum(rbf_kernel_mat))
        # print("shape kern = {0}".format(rbf_kernel_mat.shape))
        # print(y_alpha_kernel_sum)
        # print()

        alpha_sum = np.sum(alpha)
        return y_alpha_kernel_sum - alpha_sum

    def _dual_objective_function_value(self, alpha, X, y):
        # Non-vectorized
        # m = y.shape[0]
        # y_alpha_x_sum = 0
        # for i in range(m):
        #     for j in range(m):
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


        
