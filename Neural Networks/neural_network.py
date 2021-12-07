import numpy as np

class NeuralNetwork():
    def __init__(self, H):
        """ Create a trainable standard 3 fully connected layer neural network of the following form (output dimensions of layer are indicated in parenthesis):

        input X (mxd) -> fully connected layer with sigmoid activation mapping to H hidden units (mxH) -> fully connected layer with sigmoid activation mapping to H hidden units (mxH) -> fully connected layer with no activation mapping to 1 output unit (mx1) -> output (mx1)

        Args:
            H (int): Desired hidden layer size of size greater than 0
        """
        self.H = H 
        self.W1 = np.array([]) # (dxH)
        self.b1 = np.array([]) # (H)
        self.W2 = np.array([]) # (HxH)
        self.b2 = np.array([]) # (H)
        self.W3 = np.array([]) # (Hx1)
        self.b3 = np.array([]) # (1)

    def _initialize_weights(self, d, random_weight_initialization=True):
        self.W1 = np.random.normal(size=(d,self.H)) if random_weight_initialization else np.zeros((d,self.H)) # (dxH)
        self.b1 = np.random.normal(size=(self.H)) if random_weight_initialization else np.zeros((self.H)) # (H)
        self.W2 = np.random.normal(size=(self.H,self.H)) if random_weight_initialization else np.zeros((self.H,self.H)) # (HxH)
        self.b2 = np.random.normal(size=(self.H)) if random_weight_initialization else np.zeros((self.H)) # (H)
        self.W3 = np.random.normal(size=(self.H,1)) if random_weight_initialization else np.zeros((self.H,1)) # (Hx1)
        self.b3 = np.random.normal(size=(1)) if random_weight_initialization else np.zeros((1)) # (1)
    
    def _forward_pass(self, X):
        S1 = np.dot(X, self.W1) + self.b1 # mxH
        Z1 = self._sigmoid(S1) # mxH

        S2 = np.dot(Z1, self.W2) + self.b2 # mxH
        Z2 = self._sigmoid(S2) # mxH

        score = np.dot(Z2, self.W3) + self.b3 # mx1

        cache = (S1, Z1, S2, Z2)

        return score, cache

    def _backwards_pass(self, X, y, scores, cache):
        # Backprop as written below, should work generally for batch sizes other than m=1 not just 1 example if dy is an mx1 vector
        # Note that d* implies ((partial d) / (partial *))
        S1, Z1, S2, Z2 = cache

        dy = (scores - y).reshape((1,1)) # (mx1)
        dW3 = np.dot(Z2.reshape((1,-1)).T, dy) # (mxH).T dot (mx1) => (Hxm) dot (mx1) => (Hx1)
        db3 = np.sum(dy, axis=0) # sum((mx1), axis=0) => (1)
        dZ2 = np.dot(dy, self.W3.reshape((-1,1)).T) # (mx1) dot (Hx1).T => (mx1) dot (1xH) => (mxH)

        dsigmoid_2 = self._sigmoid(S2) * (1 - self._sigmoid(S2)) * dZ2 # (mxH)*(1-(mxH))*(mxH) => (mxH)
        dW2 = np.dot(Z1.reshape((1,-1)).T, dsigmoid_2) # (mxH).T dot (mxH) => (Hxm) dot (mxH) => (HxH)
        db2 = np.sum(dsigmoid_2, axis=0) # (mxH) => (H)
        dZ1 = np.dot(dsigmoid_2, self.W2.T) # (mxH) dot (HxH).T => (mxH) dot (HxH) => (mxH)

        dsigmoid_1 = self._sigmoid(S1) * (1 - self._sigmoid(S1)) * dZ1 # (mxH)*(1-(mxH))*(mxH) => (mxH)
        dW1 = np.dot(X.reshape((1,-1)).T, dsigmoid_1) # (mxd).T dot (mxH) => (dxm) dot (mxH) => (dxH)
        db1 = np.sum(dsigmoid_1, axis=0) # (mxH) => (H)
        
        return dW1, db1, dW2, db2, dW3, db3

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def _MSE(self, scores, y):
        return 0.5 * ((scores - y)**2)

    def train(self, X, y, T=100, abs_MSE_diff_thresh=1e-6, r=0.01, random_weight_initialization=True, r_sched=None):
        """ Train a neural network from scratch using stochastic gradient decsent with MSE as loss. Will set the weights of this object to the values found from training.

        Args:
            X (ndarray): Feature data. An m by d shape 2D numpy array where m is the number of examples and d is the number of features. 
            y (ndarray): Binary (1 or -1) label data. A m shape 1D numpy array where m is the number of examples.
            T (int, optional): Number of epochs to train the neural network on over the entire set of examples. Defaults to 10.
            abs_MSE_diff_thresh (float, optional): The threshold to stop stochastic gradient descent (what is considered convergence), the absolute value between the current and previous MSE loss values on given training data. Defaults to 1e-6.
            r (float, optional): Learning rate. Defaults to 0.01.
            r_sched (ndarray, optional): Learning rate schedule for each epoch (will override usage of r since r for each epoch was given). Defaults to None which means not learning rate schedule is used and constant r is used for each epoch.

        Returns:
            list: Returns a list containing the MSE loss on the training data at each epoch. 
        """
        m, d = X.shape
        self._initialize_weights(d, random_weight_initialization=random_weight_initialization)

        example_idxs_shuffle = np.arange(m)
        current_MSE = 0
        MSE_by_epoch = []
        for t in range(T):
            np.random.shuffle(example_idxs_shuffle) # Shuffle in-place
            r_t = r if r_sched is None else r_sched[t]
            for i in example_idxs_shuffle:
                x = X[i,:].reshape((1,-1)) # (1xd) 
                score, cache = self._forward_pass(x) # (1), ((1xH),(1xH),(1xH),(1xH)) => (scalar), ((H),(H),(H),(H))

                dW1, db1, dW2, db2, dW3, db3 = self._backwards_pass(x, y[i], score, cache)

                # Update weights
                self.W1 -= r_t * dW1 # (dxH)
                self.b1 -= r_t * db1 # (H)
                self.W2 -= r_t * dW2 # (HxH)
                self.b2 -= r_t * db2 # (H)
                self.W3 -= r_t * dW3 # (Hx1)
                self.b3 -= r_t * db3 # (1)

            # Calculate and update current cost
            scores, _ = self._forward_pass(X)
            new_MSE = np.mean(self._MSE(scores, y))
            abs_MSE_diff = abs(current_MSE - new_MSE)
            current_MSE = new_MSE
            MSE_by_epoch.append(current_MSE)

            # Break if the convergence critera is met
            if abs_MSE_diff < abs_MSE_diff_thresh:
                break
        return MSE_by_epoch


    def predict(self, X):
        """ Predict using a trained neural network. Note data must have same features as the data trained on.

        Args:
            X (ndarray): Feature data. An m by d shape 2D numpy array where m is the number of examples and d is the number of features. 

        Returns:
            ndarray: A n shape 1D numpy array representing the predictions with the current learned weights. 
        """
        score, _ = self._forward_pass(X)
        return np.sign(score.flatten())