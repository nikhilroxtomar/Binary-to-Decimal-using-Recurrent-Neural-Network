import numpy as np
np.random.seed(1)

def bin2int(bin_list):
    #bin_list = [0, 0, 0, 1]
    int_val = ""
    for k in bin_list:
        int_val += str(int(k))
    #int_val = 11011011    
    return int(int_val, 2)


def dataset(num):
    # num - no of samples
    bin_len = 8
    X = np.zeros((num, bin_len))
    Y = np.zeros((num))

    for i in range(num):
        X[i] = np.around(np.random.rand(bin_len)).astype(int)
        Y[i] = bin2int(X[i])
    return X, Y

no_of_smaples = 20

trainX, trainY = dataset(no_of_smaples)
testX, testY = dataset(5)

class RNN:
    def __init__(self):
        self.W = [1, 1]
        self.W_delta = [0.001, 0.001]
        self.W_sign = [0, 0]

        self.eta_p = 1.2
        self.eta_n = 0.5

    def state(self, xk, sk):
        return xk * self.W[0] + sk * self.W[1]

    def forward_states(self, X):
        S = np.zeros((X.shape[0], X.shape[1]+1))
        for k in range(0, X.shape[1]):
            next_state = self.state(X[:,k], S[:,k])
            S[:,k+1] = next_state
        return S

    def output_gradient(self, guess, real):
        return 2 * (guess - real) / no_of_smaples

    def backward_gradient(self, X, S, grad_out):
        grad_over_time = np.zeros(( X.shape[0], X.shape[1]+1 ))
        grad_over_time[:,-1] = grad_out

        wx_grad = 0
        wr_grad = 0
        for k in range(X.shape[1], 0, -1):
            wx_grad += np.sum( grad_over_time[:, k] * X[:, k-1] )
            wr_grad += np.sum( grad_over_time[:, k] * S[:, k-1] )

            grad_over_time[:, k-1] = grad_over_time[:, k] * self.W[1]
        return (wx_grad, wr_grad), grad_over_time

    def update_rprop(self, X, Y, W_prev_sign, W_delta):
        S = self.forward_states(X)
        grad_out =  self.output_gradient(S[:, -1], Y)
        W_grads, _ = self.backward_gradient(X, S, grad_out)
        self.W_sign = np.sign(W_grads)

        for i, _ in enumerate(self.W):
            if self.W_sign[i] == W_prev_sign[i]:
                W_delta[i] *= self.eta_p
            else:
                W_delta[i] *= self.eta_n
        self.W_delta = W_delta

    def train(self, X, Y, training_epochs):
        for epochs in range(training_epochs):
            self.update_rprop(X, Y, self.W_sign, self.W_delta)

            for i, _ in enumerate(self.W):
                self.W[i] -= self.W_sign[i] * self.W_delta[i]


rnn = RNN()
rnn.train(trainX, trainY, 20000)
print "Weight: \t", rnn.W
print "Real: \t\t", testY

y = rnn.forward_states(testX)[:, -1]
print "Predicted: \t",y
