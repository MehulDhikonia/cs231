import numpy as np
from random import shuffle
from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shnum_classes = W.shape[1]
  num_train = X.shape[0]ape (D, C) containing weights.
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
  
  num_classes = W.shape[1]
  num_train = X.shape[0]

  f = X.dot(W)
  #make computable
  f -= np.max(f,axis=1)[:,None]
  #exponetiate and normalize
  expf = np.exp(f) / np.sum(np.exp(f), axis=1)[:,None]
  #only considering "probabilities*" of actual class
  fi = expf[range(f.shape[0]), y]
  L = -1*np.log(fi)
  
  loss = np.sum(L)
  
  dscores = expf
  dscores[range(f.shape[0]), y] -= 1
  dW = X.T.dot(dscores)
  #normalize
  loss /= num_train
  dW /= num_train
  #regularize
  loss += reg * np.sum(W * W)
  regdW = (2*reg)*W
  dW += regdW
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  pass
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  
  num_classes = W.shape[1]
  num_train = X.shape[0]

  f = X.dot(W)
  #make computable
  f -= np.max(f,axis=1)[:,None]
  #exponetiate and normalize
  expf = np.exp(f) / np.sum(np.exp(f), axis=1)[:,None]
  #only considering "probabilities*" of actual class
  fi = expf[range(f.shape[0]), y]
  L = -1*np.log(fi)
  
  loss = np.sum(L)
  
  dscores = expf
  dscores[range(f.shape[0]), y] -= 1
  dW = X.T.dot(dscores)

  #normalize
  loss /= num_train
  dW /= num_train
  #regularize
  loss += reg * np.sum(W * W)
  regdW = (2*reg)*W
  dW += regdW
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  pass
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

