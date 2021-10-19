# Libraries
import numpy as np

class LMSRegression:
    def __init__(self):
        """ Create least mean squares regression (LMS) model that can be trained using LMS batch gradient descent, LMS stochastic gradient descent, and directly using the analytical LMS solution.
        """

        self.w = np.array([])

    def train_batch_gradient_descent(self, X, y, r=0.01, norm_w_diff_thresh=1e-6):
        """ Train using batch gradient descent. Will set the weight vector property of this object to the value found from training.

        Args:
            X (ndarray): An m by d 2D numpy array where m is the number of examples and d is the number of features. 
            y (ndarray): A m 1D numpy array where m is the number of examples.
            r (float, optional): Learning rate for gradient descent. Defaults to 0.01.
            norm_w_diff_thresh (float, optional): The threshold to stop batch gradeint descent (what is considered convergence), the norm of the difference between the current weight vector and the previous. Defaults to 1e-6.

        Returns:
            list: Returns a list containing the cost at each weight update. 
        """

        cost_vals = []
        num_examples, num_features = X.shape
        self.w = np.zeros(num_features)
        while True:
            # Less optimized gradient calculation
            # grad_J = np.zeros_like(self.w) 
            # for j in range(num_features):
            #     grad_J[j] = -np.sum((y - np.dot(X,self.w)) * X[:,j])
            # Calculate gradient
            grad_J = np.dot(X.T, -(y-np.dot(X, self.w)))
            new_w = self.w - r * grad_J

            # Update weights
            norm_w_diff = np.linalg.norm((new_w-self.w))
            self.w = new_w

            # Calculate current cost
            cost_vals.append(LMSRegression.cost(X, y, self.w))

            # Return if the convergence critera is met
            if norm_w_diff < norm_w_diff_thresh:
                return cost_vals 

    def train_stochastic_gradient_descent(self, X, y, r=0.01, abs_cost_diff_thresh=1e-6):
        """ Train using stochastic gradient descent with randomized sampling. Will set the weight vector property of this object to the value found from training.

        Args:
            X (ndarray): An m by d 2D numpy array where m is the number of examples and d is the number of features. 
            y (ndarray): A m 1D numpy array where m is the number of examples.
            r (float, optional): Learning rate for gradient descent. Defaults to 0.01.
            abs_cost_diff_thresh (float, optional): The threshold to stop stochastic gradient descent (what is considered convergence), the absolute value between the current and previous cost values. Defaults to 1e-6.

        Returns:
            list: Returns a list containing the cost at each stochastic weight update. 
        """

        cost_vals = []
        num_examples, num_features = X.shape
        self.w = np.zeros(num_features)
        example_idxs = np.arange(num_examples)
        current_cost = 0
        while True:
            for i in range(num_examples):
                # Was getting better performance (more stable convergence) without random sampling...(to not use random sampling replace random_example_idx with just i)
                # random_example_idx = i 
                random_example_idx = np.random.choice(example_idxs)
                # Update weights
                self.w += r * ((y[random_example_idx] - np.dot(self.w, X[random_example_idx,:])) * X[random_example_idx,:])

                # Calculate and update current cost
                new_cost = LMSRegression.cost(X, y, self.w)
                abs_cost_diff = abs(new_cost - current_cost)
                current_cost = new_cost
                cost_vals.append(current_cost)

                # Break if the convergence critera is met
                if abs_cost_diff < abs_cost_diff_thresh:
                    return cost_vals
    
    def train_analytical(self, X, y):
        """ Will set the weight vector property of this object to the value found from the analytical LMS solution.

        Args:
            X (ndarray): An m by d 2D numpy array where m is the number of examples and d is the number of features. 
            y (ndarray): A m 1D numpy array where m is the number of examples.
        """

        # Different ordering than lecture notes (each X is transposed) because I have X as an mxd matrix(num_examples x num_features) rather than a dxm matrix.
        self.w = np.dot((np.linalg.inv(X.T @ X) @ X.T), y)

    def predict(self, X):
        return np.dot(X, self.w)

    @classmethod
    def cost(cls, X, y, w):
        J = 0.5 * np.sum((y - np.dot(X, w))**2)
        return J

        
