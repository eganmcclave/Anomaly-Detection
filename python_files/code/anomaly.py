
# Load necessary modules
import numpy as np
from math import log

# Defined functions and classes
class IsoTree:
  def __init__(self, data):
    ''' Basic isolation tree implementation 
    Attributes:
        - l_node: either None or an IsoTree object depending on if a split is necessary
        - r_node: either None or an IsoTree object depending on if a split is necessary
    '''

    # Basic structure of tree
    self.l_node = None
    self.r_node = None
    self.num_obs = data.shape[0]
    self.split_attr = None
    self.split_val = None

    ## Training a tree is done recursively
    # Randomly select a column and a random split value from that column
    split_attr = np.random.randint(data.shape[1])
    data_attr = data[:,split_attr]
    attr_min, attr_max = np.amin(data_attr), np.amax(data_attr)
    
    # If the minimum and maximum values of this attribute are not the same
    # then we can continue to split the data
    if attr_min != attr_max:
      split_val = np.random.uniform(attr_min, attr_max)

      # Filter the data into subsets based on the split value
      data_l = data[np.where(data_attr <= split_val)[0],:]
      data_r = data[np.where(data_attr > split_val)[0],:]

      # Create new trees if necessary
      l_nonempty = data_l.shape[0] != 0
      r_nonempty = data_r.shape[0] != 0

      # Create left tree if split reveals non-unique data
      if l_nonempty:
        self.l_node = IsoTree(data_l)

      # Create right tree if split reveals non-unique data
      if r_nonempty:
        self.r_node = IsoTree(data_r)

      # Add attributes if there is a split made 
      if l_nonempty or r_nonempty:
        self.split_attr = split_attr
        self.split_val = split_val

  def evaluate(self, x, hlim, e=0):
    ''' Evaluate the path length of x for a given tree
    Input:
      - x (np array): np array containing a single observation
      - hlim (int): height limit for recursive depth
      - e (int): current path length
    Output:
      - path_len (int): length of path from root of iTree
    '''

    # Determine the path length by seeing if the current node goes beyond the 
    # height limit or is a leaf node then return the current path length and 
    # some adjustment score; otherwise continue evaluating the path length
    if hlim < e or self.l_node is None or self.r_node is None:
      return e + adjust_score(self.num_obs)
    else:
      if x[self.split_attr] <= self.split_val:
        return self.l_node.evaluate(x, hlim, e+1)
      else:
        return self.r_node.evaluate(x, hlim, e+1)

class IsoForest:
  def __init__(self, X, num_trees=100, sample_size=256):
    ''' Basic isolation forest implementation 
    Attributes:
        - shape (tuple): tuple containing the dimensionality of X, it should
            be 3 dimensional representing tile_id x frames x features
        - trees (list): list of num_trees many IsoTree objects 
    '''

    # Basic structure of a forest
    self.shape = X.shape
    self.trees = []

    ## Training a forest is done iteratively 
    # Build a collection of trees for a given forest
    for _ in range(num_trees):
      X_sub = X[np.random.choice(self.shape[0], sample_size),:]
      self.trees.append(IsoTree(X_sub))

  def anomaly_score(self, x, hlim):
    ''' Calculate the anomaly score for a single instance of x in an IsoForest
    Input:
      - x (np array): np array containing a single observation
      - hlim (int): height limit for recursive depth path evaluation
    Output:
      - s_score (float): the anomaly score for x across all IsoTrees
    '''
    
    scores = []
    for iTree in self.trees:
      scores.append(iTree.evaluate(x, hlim))
    s_score = 2 ** (-np.mean(scores) / adjust_score(self.shape[0]))

    return s_score

  def find_anomalies(self, X_new, hlim):
    ''' Calculates the anomaly score for every instance of the new data
    Input:
      - X_new (np array): np array which contains data
      - hlim (int): height limit for recursive depth path evaluation
    Output:
      - s_scores (list): list of anomaly scores for all x in X_new
    '''

    s_scores = []
    for x in X_new:
      s_scores.append(self.anomaly_score(x, hlim))

    return s_scores

def adjust_score(obs):
  ''' Determine the adjustment score given a sample size
  Input:
    - obs (int): the number of total observations in the full dataset
  Output:
    - c_score (double): the adjustment score to be used for isoTree evaluation
  '''

  c_score = 0
  if obs > 2:
    c_score = 2 * (log(obs - 1) + 0.5772156649) - 2 * (obs - 1) / (obs)
  elif obs == 2:
    c_score = 1

  return c_score

