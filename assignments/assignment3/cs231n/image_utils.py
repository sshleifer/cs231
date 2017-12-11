import urllib2, os, tempfile
import numpy as np

from cs231n.fast_layers import conv_forward_fast


"""
Utility functions used for viewing and processing images.
"""


def blur_image(X):
  """
  A very gentle image blurring operation, to be used as a regularizer for image
  generation.
  
  Inputs:
  - X: Image data of shape (N, 3, H, W)
  
  Returns:
  - X_blur: Blurred version of X, of shape (N, 3, H, W)
  """
  w_blur = np.zeros((3, 3, 3, 3))
  b_blur = np.zeros(3)
  blur_param = {'stride': 1, 'pad': 1}
  for i in xrange(3):
    w_blur[i, i] = np.asarray([[1, 2, 1], [2, 188, 2], [1, 2, 1]], dtype=np.float32)
  w_blur /= 200.0
  return conv_forward_fast(X, w_blur, b_blur, blur_param)[0]


def preprocess_image(img, mean_img, mean='image'):
  """
  Convert to float, transepose, and subtract mean pixel
  
  Input:
  - img: (H, W, 3)
  
  Returns:
  - (1, 3, H, 3)
  """
  if mean == 'image':
    mean = mean_img
  elif mean == 'pixel':
    mean = mean_img.mean(axis=(1, 2), keepdims=True)
  elif mean == 'none':
    mean = 0
  else:
    raise ValueError('mean must be image or pixel or none')
  return img.astype(np.float32).transpose(2, 0, 1)[None] - mean


def deprocess_image(img, mean_img, mean='image', renorm=False):
  """
  Add mean pixel, transpose, and convert to uint8
  
  Input:
  - (1, 3, H, W) or (3, H, W)
  
  Returns:
  - (H, W, 3)
  """
  if mean == 'image':
    mean = mean_img
  elif mean == 'pixel':
    mean = mean_img.mean(axis=(1, 2), keepdims=True)
  elif mean == 'none':
    mean = 0
  else:
    raise ValueError('mean must be image or pixel or none')
  if img.ndim == 3:
    img = img[None]
  img = (img + mean)[0].transpose(1, 2, 0)
  if renorm:
    low, high = img.min(), img.max()
    img = 255.0 * (img - low) / (high - low)
  return img.astype(np.uint8)


def image_from_url(url):
  """
  Read an image from a URL. Returns a numpy array with the pixel data.
  We write the image to a temporary file then read it back. Kinda gross.
  """
  from scipy.misc import imread
  try:
    f = urllib2.urlopen(url)
    _, fname = tempfile.mkstemp()
    with open(fname, 'wb') as ff:
      ff.write(f.read())
    img = imread(fname)
    os.remove(fname)
    return img
  except urllib2.URLError as e:
    print 'URL Error: ', e.reason, url
  except urllib2.HTTPError as e:
    print 'HTTP Error: ', e.code, url


def shower(t):
  plt.gcf().set_size_inches(3, 3)
  plt.axis('off')
  plt.title('t = %d' % t)
  plt.show()

import matplotlib.pyplot as plt
def invert_features_annealed(data, target_feats, layer, model, learning_rate=1e5, num_iterations=500,
                    l2_reg=1e7, blur_every=1, show_every=50, X=None, **kwargs):
  """
  Perform feature inversion in the style of Mahendran and Vedaldi 2015, using
  L2 regularization and periodic blurring.

  Inputs:
  - target_feats: Image features of the target image, of shape (1, C, H, W);
  we will try to generate an image that matches these features
  - layer: The index of the layer from which the features were extracted
  - model: A PretrainedCNN that was used to extract features

  Keyword arguments:
  - learning_rate: The learning rate to use for gradient descent
  - num_iterations: The number of iterations to use for gradient descent
  - l2_reg: The strength of L2 regularization to use; this is lambda in the
  equation above.
  - blur_every: How often to blur the image as implicit regularization; set
  to 0 to disable blurring.
  - show_every: How often to show the generated image; set to 0 to disable
  showing intermediate reuslts.

  Returns:
  - X: Generated image of shape (1, 3, 64, 64) that matches the target features.
  """
  if not isinstance(X, np.ndarray):
    X = np.random.randn(1, 3, 64, 64)
  last_improved_X = X.copy()
  logs = []
  shown = []
  last_score = np.inf
  for t in xrange(num_iterations):
    out, cache = model.forward(X, end=layer)
    errors = np.linalg.norm(out - target_feats) ** 2

    dX, grads = model.backward(errors, cache)
    mean_reconstruction_error = np.mean(errors)
    reg_loss = (X ** 2) * l2_reg
    d_reg_loss = X * l2_reg * 2

    optimal_update = (dX - d_reg_loss) * learning_rate
    logs.append({'reg_loss_mean': np.abs(reg_loss).mean(),
       'abs_update_mean': np.abs(optimal_update).mean(),
       'avg_error': mean_reconstruction_error,
       'max_update': np.abs(optimal_update).max(),
                 'learning_rate': learning_rate})
    X += optimal_update
    # As a regularizer, clip the image
    X = np.clip(X, -data['mean_image'], 255.0 - data['mean_image'])
    if (mean_reconstruction_error + np.mean(reg_loss)) > last_score:
      if learning_rate <= 3e-4:
        break
      learning_rate = max(learning_rate * .1, 3e-4)
      print 'Got worse at t={}, decreasing lr to {}'.format(t, learning_rate)
      X = last_improved_X
      continue
    else:
      last_score = mean_reconstruction_error
      last_improved_X = X

    # As a regularizer, periodically blur the image
    #if (blur_every > 0) and t % blur_every == 0:
    #  X = blur_image(X)

    if (show_every > 0) and (t % show_every == 0 or t + 1 == num_iterations):
      depro = deprocess_image(X, data['mean_image'])
      plt.imshow(depro)
      shower(t)
      shown.append(depro)

  return X, logs, shown
