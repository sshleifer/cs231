import numpy as np

from cs231n.layers import *
from cs231n.layer_utils import *

np.random.seed(42)

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
    self.W1 = np.random.randn(input_dim, hidden_dim) * weight_scale
    self.b1 = np.zeros(hidden_dim)
    self.W2 = np.random.randn(hidden_dim, num_classes) * weight_scale
    self.b2 = np.zeros(num_classes)
    self.params = {'W1': self.W1, 'W2': self.W2, 'b1': self.b1, 'b2': self.b2}

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
    x, cache1 = affine_relu_forward(X, self.params['W1'],  self.params['b1'])
    scores, cache2 = affine_forward(x, self.params['W2'], self.params['b2'])

    if y is None: # test mode
      return scores
    loss, dout = softmax_loss(scores, y)
    grads = {}
    reg_losses = [np.sum(v * v) for k, v in self.params.items()
                 if k in ['W1', 'W2']]
    reg_loss = np.sum(reg_losses) * self.reg * .5
    loss += reg_loss
    dx, dw2, db2 = affine_backward(dout, cache2)
    grads = dict(W2=dw2 + self.reg * self.params['W2'], b2=db2)
    _, dw1, db1 = affine_relu_backward(dx, cache1)
    grads.update(dict(W1=dw1 + self.reg * self.params['W1'], b1=db1))
    return loss, grads


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
    def scaled_random_layer(in_dim, out_dim):
        return np.random.randn(in_dim, out_dim) * weight_scale
    self.use_batchnorm = use_batchnorm
    self.use_dropout = dropout > 0
    self.reg = reg
    self.num_layers = 1 + len(hidden_dims)
    self.dtype = dtype
    self.params = {}
    self.bn_params = []
    self.params['W0'] = scaled_random_layer(input_dim, hidden_dims[0])
    self.params['b0'] = np.zeros(hidden_dims[0])
    if use_batchnorm:
        self.params['gamma0'] = np.ones(hidden_dims[0])
        self.params['beta0'] = np.zeros(hidden_dims[0])
        self.bn_params.append({'mode': 'train'})

    for i, in_dim in enumerate(hidden_dims):
        idx = i + 1
        out_dim = num_classes if idx == (self.num_layers - 1) else hidden_dims[i+1]
        self.params['W{}'.format(idx)] = scaled_random_layer(in_dim, out_dim)
        self.params['b{}'.format(idx)] = np.zeros(out_dim)
        if use_batchnorm:
           self.params['gamma{}'.format(idx)] = np.ones(out_dim) #(in_dim, out_dim)
           self.params['beta{}'.format(idx)] = np.ones(out_dim)
           self.bn_params.append({'mode': 'train'})
    # When using dropout we need to pass a dropout_param dictionary to each
    # dropout layer so that the layer knows the dropout probability and the mode
    # (train / test). You can pass the same dropout_param to each dropout layer.
    self.dropout_param = {}
    if self.use_dropout:
      self.dropout_param = {'mode': 'train', 'p': dropout}
      if seed is not None:
        self.dropout_param['seed'] = seed
    # self.bn_params = []
    # if self.use_batchnorm:
    #   self.bn_params = [{'mode': 'train'} for i in xrange(self.num_layers - 1)]


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
    x = X.copy()
    cache_dict = {}
    dropout_cache = {}
    bn_cache = {}

    for i, layer in enumerate(range(self.num_layers)):
        if layer == self.num_layers - 1:
            forward_func = affine_forward
        else:
            forward_func = affine_relu_forward
        x, cache = forward_func(x,
                                self.params['W{}'.format(i)],
                                self.params['b{}'.format(i)])
        if self.use_dropout:
            x, do_cache = dropout_forward(x, self.dropout_param)
            dropout_cache[layer] = do_cache
        if self.use_batchnorm and layer <= len(self.bn_params):
            g, b = (self.params['gamma{}'.format(layer)], self.params['beta{}'.format(layer)])

            x, bcache = batchnorm_forward(x, g, b, self.bn_params[layer])
            bn_cache[layer] = bcache
        cache_dict[layer] = cache
    scores = x
                                                         #
    # When using dropout, you'll need to pass self.dropout_param to each       #
    # dropout forward pass.                                                    #
    #                                                                          #
    # When using batch normalization, you'll need to pass self.bn_params[0] to #
    # the forward pass for the first batch normalization layer, pass           #
    # self.bn_params[1] to the forward pass for the second batch normalization #
    # layer, etc.                                                              #
    # If test mode return early
    if mode == 'test':
      return scores

    loss, dout = softmax_loss(scores, y)
    grads = {}
    reg_losses = [np.sum(v * v) for k, v in self.params.items()
                  if k.startswith('W')]
    reg_loss = np.sum(reg_losses) * self.reg * .5
    loss += reg_loss
    grads = {}
    dx = dout
    for i, layer in enumerate(reversed(range(self.num_layers))):
        if layer == self.num_layers - 1:
            backward_func = affine_backward
        else:
            backward_func = affine_relu_backward
        if self.use_batchnorm and layer in bn_cache:
            dx, dgamma, dbeta = batchnorm_backward(dx, bn_cache[layer])
            grads['gamma{}'.format(layer)] = dgamma
            grads['beta{}'.format(layer)] = dbeta
        if self.use_dropout and layer in dropout_cache:
            dx = dropout_backward(dx, dropout_cache[layer])
        dx, dw, db = backward_func(dx, cache_dict[layer])
        grads['W{}'.format(layer)] = dw + self.reg * self.params['W{}'.format(layer)]
        grads['b{}'.format(layer)] = db
    # loss in the loss variable and gradients in the grads dictionary. Compute #
    # data loss using softmax, and make sure that grads[k] holds the gradients #
    # for self.params[k]. Don't forget to add L2 regularization!               #
    #                                                                          #
    # When using batch normalization, you don't need to regularize the scale   #
    # and shift parameters.                                                    #
    #                                                                          #
    return loss, grads
