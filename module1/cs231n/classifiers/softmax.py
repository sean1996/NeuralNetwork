import numpy as np
from random import shuffle

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
  num_train = X.shape[0]
  num_classes = W.shape[1]
  num_dim = X.shape[1]


  for row in range(num_train):
    scores = X[row].dot(W)    #scores has dim (1, 10)

    #shifting values in scores by the maximum element in the array to deal with numeric instability
    log_c = np.max(scores)
    scores -= log_c
    sum_exp_scores = np.sum(np.exp(scores))

    #compute loss of this training example
    loss +=  -1 * np.log(np.exp(scores[y[row]]) / sum_exp_scores) #here row denotes ith training example

    #compute gradients
    for col in range(num_classes):
      p = np.exp(scores[col])/sum_exp_scores
      dW[:, col] += (p-(col == y[row])) * X[row, :]

  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  #Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  dW /= num_train
  loss /= num_train

  #Regulization
  dW += reg*W
  loss +=  0.5* reg * np.sum(W * W)

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
  num_classes = W.shape[1]
  num_train = X.shape[0]
  scores = X.dot(W)

  #compute softmax probability output using fully vectorized operation
  shift_scores = scores - np.max(scores, axis = 1).reshape(-1,1)
  softmax_output = np.exp(shift_scores)/np.sum(np.exp(shift_scores), axis = 1).reshape(-1,1)

  #shifting values in scores by the maximum element in the array to deal with numeric instability
  loss = -np.sum(np.log(softmax_output[range(num_train), list(y)]))

  #average
  loss /= num_train 
  loss +=  0.5* reg * np.sum(W * W)
  
  dS = softmax_output.copy()
  dS[range(num_train), list(y)] += -1
  dW = (X.T).dot(dS)
  dW = dW/num_train + reg* W 
  #pass
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

