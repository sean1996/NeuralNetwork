import numpy as np

from cs231n.layers import *
from cs231n.layer_utils import *


class TwoLayerNet(object):
  """
  A two-layer fully-connected neural network with ReLU nonlinearity and
  softmax loss that uses a modular layer design. We assume an input dimension
  of D, a hidden dimension of H, and perform classification over C classes.
  
  The architecure should be affine - relu - affine - softmax.

  Note that this class does not implement gradient descent; instead, it
  will interact with a separate Solver object that is responsible for running
  optimization.

  The learnable parameters of the model are stored in the dictionary
  self.params that maps parameter names to numpy arrays.
  """
  
  def __init__(self, input_dim=3*32*32, hidden_dim=100, num_classes=10,
               weight_scale=1e-3, reg=0.0):
    """
    Initialize a new network.

    Inputs:
    - input_dim: An integer giving the size of the input
    - hidden_dim: An integer giving the size of the hidden layer
    - num_classes: An integer giving the number of classes to classify
    - dropout: Scalar between 0 and 1 giving dropout strength.
    - weight_scale: Scalar giving the standard deviation for random
      initialization of the weights.
    - reg: Scalar giving L2 regularization strength.
    """
    self.params = {}
    self.reg = reg
    
    ############################################################################
    # TODO: Initialize the weights and biases of the two-layer net. Weights    #
    # should be initialized from a Gaussian with standard deviation equal to   #
    # weight_scale, and biases should be initialized to zero. All weights and  #
    # biases should be stored in the dictionary self.params, with first layer  #
    # weights and biases using the keys 'W1' and 'b1' and second layer weights #
    # and biases using the keys 'W2' and 'b2'.                                 #
    ############################################################################
    self.params['W1'] = weight_scale * np.random.randn(input_dim, hidden_dim)
    self.params['b1'] = np.zeros(hidden_dim)
    self.params['W2'] = weight_scale * np.random.randn(hidden_dim, num_classes)
    self.params['b2'] = np.zeros(num_classes)
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################


  def loss(self, X, y=None):
    """
    Compute loss and gradient for a minibatch of data.

    Inputs:
    - X: Array of input data of shape (N, d_1, ..., d_k)
    - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

    Returns:
    If y is None, then run a test-time forward pass of the model and return:
    - scores: Array of shape (N, C) giving classification scores, where
      scores[i, c] is the classification score for X[i] and class c.

    If y is not None, then run a training-time forward and backward pass and
    return a tuple of:
    - loss: Scalar value giving the loss
    - grads: Dictionary with the same keys as self.params, mapping parameter
      names to gradients of the loss with respect to those parameters.
    """  
    scores = None
    ############################################################################
    # TODO: Implement the forward pass for the two-layer net, computing the    #
    # class scores for X and storing them in the scores variable.              #
    ############################################################################
    out_first_second_layer, cache_first_second_layer = affine_relu_forward(X, self.params['W1'], self.params['b1'])
    out_third_layer, cache_third_layer = affine_forward(out_first_second_layer, self.params['W2'], self.params['b2'])
    scores = out_third_layer
    #QUESTION: why don't we need to add softmax function in the forth layer here?????

    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    # If y is None then we are in test mode so just return scores
    if y is None:
      return scores
    
    loss, grads = 0, {}
    ############################################################################
    # TODO: Implement the backward pass for the two-layer net. Store the loss  #
    # in the loss variable and gradients in the grads dictionary. Compute data #
    # loss using softmax, and make sure that grads[k] holds the gradients for  #
    # self.params[k]. Don't forget to add L2 regularization!                   #
    #                                                                          #
    # NOTE: To ensure that your implementation matches ours and you pass the   #
    # automated tests, make sure that your L2 regularization includes a factor #
    # of 0.5 to simplify the expression for the gradient.                      #
    ############################################################################
    
    #compute loss and gradient for softmax function 
    loss_softmax, grad_softmax = softmax_loss(scores, y)

    #normalization
    loss_softmax = loss_softmax + 0.5 * self.reg * np.sum(self.params['W1'] * self.params['W1']) + 0.5 * self.reg * np.sum(self.params['W2'] * self.params['W2'])

    #Use back propagation to compute the gradient of loss function with regards to X, W1, W2, b1, b2
    #Start from the gradient that we stored in grad_softmax in the last output layer, this is the gradient of loss regarding to input of softmax layer
    dx2, dw2, db2 = affine_backward(grad_softmax, cache_third_layer)
    grads['W2'] = dw2 + self.reg * self.params['W2']
    grads['b2'] = db2
    dx1, dw1, db1 = affine_relu_backward(dx2, cache_first_second_layer)
    grads['W1'] = dw1 + self.reg * self.params['W1']
    grads['b1'] = db1


    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    return loss_softmax, grads


class FullyConnectedNet(object):
  """
  A fully-connected neural network with an arbitrary number of hidden layers,
  ReLU nonlinearities, and a softmax loss function. This will also implement
  dropout and batch normalization as options. For a network with L layers,
  the architecture will be
  
  {affine - [batch norm] - relu - [dropout]} x (L - 1) - affine - softmax
  
  where batch normalization and dropout are optional, and the {...} block is
  repeated L - 1 times.
  
  Similar to the TwoLayerNet above, learnable parameters are stored in the
  self.params dictionary and will be learned using the Solver class.
  """

  def __init__(self, hidden_dims, input_dim=3*32*32, num_classes=10,
               dropout=0, use_batchnorm=False, reg=0.0,
               weight_scale=1e-2, dtype=np.float32, seed=None):
    """
    Initialize a new FullyConnectedNet.
    
    Inputs:
    - hidden_dims: A list of integers giving the size of each hidden layer.
    - input_dim: An integer giving the size of the input.
    - num_classes: An integer giving the number of classes to classify.
    - dropout: Scalar between 0 and 1 giving dropout strength. If dropout=0 then
      the network should not use dropout at all.
    - use_batchnorm: Whether or not the network should use batch normalization.
    - reg: Scalar giving L2 regularization strength.
    - weight_scale: Scalar giving the standard deviation for random
      initialization of the weights.
    - dtype: A numpy datatype object; all computations will be performed using
      this datatype. float32 is faster but less accurate, so you should use
      float64 for numeric gradient checking.
    - seed: If not None, then pass this random seed to the dropout layers. This
      will make the dropout layers deteriminstic so we can gradient check the
      model.
    """
    self.use_batchnorm = use_batchnorm
    self.use_dropout = dropout > 0
    self.reg = reg
    self.num_layers = 1 + len(hidden_dims)
    self.dtype = dtype
    self.params = {}

    ############################################################################
    # TODO: Initialize the parameters of the network, storing all values in    #
    # the self.params dictionary. Store weights and biases for the first layer #
    # in W1 and b1; for the second layer use W2 and b2, etc. Weights should be #
    # initialized from a normal distribution with standard deviation equal to  #
    # weight_scale and biases should be initialized to zero.                   #
    #                                                                          #
    # When using batch normalization, store scale and shift parameters for the #
    # first layer in gamma1 and beta1; for the second layer use gamma2 and     #
    # beta2, etc. Scale parameters should be initialized to one and shift      #
    # parameters should be initialized to zero.                                #
    ############################################################################
    current_input_dim = input_dim

    #Batch norm would apply to the input layer even before the input data is feeded into first relu layer
    if self.use_batchnorm:
        self.params['gamma%d' %1] = np.ones(input_dim)
        self.params['beta%d' %1] = np.zeros(input_dim)

    for index, current_hidden_dim in enumerate(hidden_dims):
      #initialize gammma and beta for batch norm, tricky: the gamma and betta size of layer 2 is the size of hidden neurons in layer 1 RELU
      if self.use_batchnorm and index + 2 != self.num_layers:
        self.params['gamma%d' %(index + 2)] = np.ones(current_hidden_dim)
        self.params['beta%d' %(index + 2)] = np.zeros(current_hidden_dim)
      #initizalize W and b for (N-1) relu layers
      self.params['W' + str(index + 1)] = weight_scale * np.random.randn(current_input_dim, current_hidden_dim)
      self.params['b' + str(index + 1)] = weight_scale * np.zeros(current_hidden_dim)
      current_input_dim = current_hidden_dim


    #intialize the W and b for final affine layer
    self.params['W' + str(self.num_layers)] = weight_scale * np.random.randn(current_input_dim, num_classes)
    self.params['b' + str(self.num_layers)] = weight_scale * np.zeros(num_classes)

    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    # When using dropout we need to pass a dropout_param dictionary to each
    # dropout layer so that the layer knows the dropout probability and the mode
    # (train / test). You can pass the same dropout_param to each dropout layer.
    self.dropout_param = {}
    if self.use_dropout:
      self.dropout_param = {'mode': 'train', 'p': dropout}
      if seed is not None:
        self.dropout_param['seed'] = seed
    
    # With batch normalization we need to keep track of running means and
    # variances, so we need to pass a special bn_param object to each batch
    # normalization layer. You should pass self.bn_params[0] to the forward pass
    # of the first batch normalization layer, self.bn_params[1] to the forward
    # pass of the second batch normalization layer, etc.
    self.bn_params = []
    if self.use_batchnorm:
      self.bn_params = [{'mode': 'train'} for i in xrange(self.num_layers - 1)]
    
    # Cast all parameters to the correct datatype
    for k, v in self.params.iteritems():
      self.params[k] = v.astype(dtype)


  def loss(self, X, y=None):
    """
    Compute loss and gradient for the fully-connected net.

    Input / output: Same as TwoLayerNet above.
    """
    X = X.astype(self.dtype)
    mode = 'test' if y is None else 'train'

    # Set train/test mode for batchnorm params and dropout param since they
    # behave differently during training and testing.
    if self.dropout_param is not None:
      self.dropout_param['mode'] = mode   
    if self.use_batchnorm:
      for bn_param in self.bn_params:
        bn_param[mode] = mode

    scores = None
    ############################################################################
    # TODO: Implement the forward pass for the fully-connected net, computing  #
    # the class scores for X and storing them in the scores variable.          #
    #                                                                          #
    # When using dropout, you'll need to pass self.dropout_param to each       #
    # dropout forward pass.                                                    #
    #                                                                          #
    # When using batch normalization, you'll need to pass self.bn_params[0] to #
    # the forward pass for the first batch normalization layer, pass           #
    # self.bn_params[1] to the forward pass for the second batch normalization #
    # layer, etc.                                                              #
    ############################################################################

    cacheList = [] 
    cacheList_dropout = [] 
    cacheList_batch = []
    current_layer_input = X
    #compute forward pass for (L - 1) affine-relu layers 
    for i in range(self.num_layers - 1):
      if self.use_batchnorm:
        current_layer_input, cache_batchNorm = batchnorm_forward(current_layer_input, self.params['gamma%d' %(i + 1)], self.params['beta%d' %(i + 1)], self.bn_params[i])
        cacheList_batch.append(cache_batchNorm)

      #compute forward pass for affine-relu layer and store layer cache
      current_layer_output, current_cache = affine_relu_forward(current_layer_input, self.params['W' + str(i + 1)], self.params['b' + str(i + 1)])
      cacheList.append(current_cache)
      #apply dropout and store the dropout cache, dropout is only applied to RELU layers but not the last affine linear layer
      if self.use_dropout:
        current_layer_output, cache_dropout = dropout_forward(current_layer_output, self.dropout_param)
        cacheList_dropout.append(cache_dropout)

      current_layer_input = current_layer_output
    #compute forward pass for the last affine layer
    scores, current_cache = affine_forward(current_layer_input, self.params['W' + str(self.num_layers)], self.params['b' + str(self.num_layers)])
    cacheList.append(current_cache)

    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    # If test mode return early
    if mode == 'test':
      return scores

    loss, grads = 0.0, {}
    ############################################################################
    # TODO: Implement the backward pass for the fully-connected net. Store the #
    # loss in the loss variable and gradients in the grads dictionary. Compute #
    # data loss using softmax, and make sure that grads[k] holds the gradients #
    # for self.params[k]. Don't forget to add L2 regularization!               #
    #                                                                          #
    # When using batch normalization, you don't need to regularize the scale   #
    # and shift parameters.                                                    #
    #                                                                          #
    # NOTE: To ensure that your implementation matches ours and you pass the   #
    # automated tests, make sure that your L2 regularization includes a factor #
    # of 0.5 to simplify the expression for the gradient.                      #
    ############################################################################
    #compute the loss using softmax
    loss, dscores = softmax_loss(scores, y)
    #compute loss regulation on this last layer
    loss += 0.5 * self.reg * np.sum(self.params['W' + str(self.num_layers)] * self.params['W' + str(self.num_layers)])

    #compute the gradient of last layer with regarding to loss
    dx, dw, db = affine_backward(dscores, cacheList[self.num_layers - 1])
    grads['W' + str(self.num_layers)] = dw + self.reg * self.params['W' + str(self.num_layers)]
    grads['b' + str(self.num_layers)] = db
    dout_current = dx
    #compute the gradient of all (L-1) affine-relu layers
    for i in range(self.num_layers - 2, -1, -1):
      array_index = i
      #L2 regulation
      loss += 0.5 * self.reg * np.sum(self.params['W' + str(array_index + 1)] * self.params['W' + str(array_index + 1)])
      #compute the gradient of the dropout layer on this sandwitch layer
      if self.use_dropout:
        dout_current = dropout_backward(dout_current, cacheList_dropout[array_index])

      #compute loss regulation on this sandwitch layer
      dx, dw, db = affine_relu_backward(dout_current, cacheList[array_index])

      if self.use_batchnorm:
        dx, dgamma, dbeta = batchnorm_backward(dx, cacheList_batch[array_index])
      grads['W' + str(array_index + 1)] = dw + self.reg * self.params['W' + str(array_index + 1)]
      grads['b' + str(array_index + 1)] = db
      if self.use_batchnorm:
        grads['gamma%d'%(array_index+1)] = dgamma
        grads['beta%d'%(array_index+1)] = dbeta

      dout_current = dx

    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    return loss, grads
