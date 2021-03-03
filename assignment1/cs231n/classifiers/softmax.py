from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    D, C = W.shape
    N = X.shape[0]

    S = np.matmul(X, W)
    for i in range(N):
      L = np.exp(S[i])
      softmax = L / np.sum(L)
      loss += -1.0 * np.log(softmax[y[i]])
      y_i = np.eye(C)[y[i]]

      dW += np.matmul(np.array([X[i]]).T, np.array([softmax - y_i]))

    loss /= N
    loss += reg * np.sum(np.multiply(W, W))

    dW /= N
    dW += 2.0 * reg * W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    D, C = W.shape
    N = X.shape[0]

    S = np.matmul(X, W)
    L = np.exp(S)
    softmax = (L.T / np.sum(L, axis=1)).T

    Y = np.eye(C)[y]
    softmax_Yi = np.multiply(softmax, Y)
    softmax_Yi[softmax_Yi==0] = 1 # because log(0) = inf
    loss = np.sum(-1.0 * np.log(softmax_Yi)) / N

    reg_term = reg * np.sum(np.multiply(W, W))
    loss += reg_term

    dW = np.matmul(X.T, softmax - Y) / N
    dW += 2.0 * reg * W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
