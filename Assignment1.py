# PIEN PIETERSE (6763758) 
# j.s.f.pieterse@students.uu.nl 

# JESSE HOITING (4443306) 
# j.t.hoiting@students.uu.nl

#SHIQI CHEN (4421698) 
# s.chen26@students.uu.nl



import numpy as np
import collections
import pandas as pd
import random
from collections import Counter
from scipy import stats


class DecisionNode():
    '''
    This is our Decision tree object
    '''
    def __init__(self, feature=None, splitpoint=None, left=None, right=None, value=None, leaf=False):
      '''
      optional input values for:
      - feature (int: column index) = the feature the node splits on.
      - splitpoint (float) = the value of the feature the node splits on.
      - left (DecisionNode) = the left side of the tree (points to another DecisionNode object)
      - right (DecisionNode) = the right side of the tree
      - value (np array (int)) = the predicted values in the node
      - leaf (boolean) = whether the node is a leaf or not
      returns:
      - a DecisionNode object
      '''
      self.feature = feature          # Feature to split on, column index
      self.splitpoint = splitpoint    # Threshold value for the split
      self.left = left                # Left subtree
      self.right = right              # Right subtree
      self.value = value              # If leaf node, the predicted class
      self.leaf = leaf                # If node is leaf node

    def is_leaf_node(self):
      return self.leaf


def impurity(a):
  '''
  Calculates the gini impurity of an array.
  input value:
  - a (np array (float)): the array
  returns:
  - a float
  '''
  i = (np.count_nonzero(a == 0)/len(a))*(1-np.count_nonzero(a == 0)/len(a))
  return i


def bestsplit(x,y, minleaf):
  '''
  This function calculates the best splitpoint for a single feature.
  input:
  - x (1d np array (float)): the x values of the feature
  - y (1d np array (int)): the y values of the feature
  - minleaf (int): the minimum number of observations required for a leaf node
  returns: 
  - the best splitpoint value that that feature should be split on (float)
  - the best gini reduction for that splitpoint (float)
  '''

  #The best split is the split that achieves the highest impurity reduction.
  impurity_parent = impurity(y)

  x_sorted = np.sort(np.unique(x))
  x_splitpoints = (x_sorted[0:(len(x_sorted)-1)]+x_sorted[1:(len(x_sorted))])/2

  reductions = []
  for s in x_splitpoints:
    #go over all the splitpoints
    left = impurity(y[x <= s])
    l = len(y[x <= s])/len(y)
    right = impurity(y[x > s])
    r = len(y[x > s])/len(y)

    #calculate gini reduction
    reduction = impurity_parent - ((l*left) + r*(right))

    #only if minleaf parameter is correct
    if (len(y[x<=s]) >= minleaf) and (len(y[x>s]) >= minleaf):
      reductions.append(reduction)

  # the reductions list is empty if the minleaf constraint is not satisfied
  if reductions == []:
    return None, 0

  best_splitpoint = x_splitpoints[reductions.index(max(reductions))]

  return best_splitpoint, max(reductions)


def split(x,y,minleaf,nfeat):
  '''
  This function finds which feature has the best split, using the best_split function
  input:
  - x (2d np array (float)): the x values for all the features
  - y (2d np array (int)): the y values for all the features
  - minleaf (int): the minimum number of observations required for a leaf node
  - nfeat (int): the number of features that should be considered for each split (chosen randomly)
  returns:
  - The best splitpoint for one of the features (float)
  - The column index (feature) which has the best split
  '''

  # which feature do we split on, and what is the threshold?
  best_reduction = 0
  best_splitpoint = None
  column_i = None
  i = 0
  n_features = len(x.T)

  #which random features we split on
  random_features = random.sample(range(0, n_features), nfeat)

  #go over each feature and find the best splitpoint. If a better splitpoint is found, store those values.
  for index in random_features:
    splitpoint, reduction = bestsplit(x[:, index],y,minleaf)

    if reduction > best_reduction:
      best_reduction = reduction
      best_splitpoint = splitpoint
      column_i = index
    i+=1
  return best_splitpoint, column_i


def tree_grow(x, y, nmin, minleaf, nfeat):
  '''
  A recursive function that uses the splitfunction to find the best split for the features and uses that split 
  to recursively call the tree_grow function again to build the left and right side of the tree. The function stops
  when either one of the nodes if pure, the nmin constraint or the minleaf constraint is satisfied.
  input:
  - x (2d np array (float)): the x values for all the features
  - y (2d np array (int)): the y values for all the features
  - nmin (int): the number of observations that a node must contain at least, for it to be allowed to be split
  - minleaf (int): the minimum number of observations required for a leaf node
  - nfeat (int): the number of features that should be considered for each split (chosen randomly)
  returns:
  - a DecisionNode object
  '''
  observations = len(x)

  #the number of observations that a node must contain at least, for it to be
  #allowed to be split
  #also if the node is already pure, splitting is not needed anymore
  if observations < nmin or np.all(y == y[0]): #is np.all(y == 0) correct?
    return DecisionNode(value=y, leaf=True)

  #find the column that has the best quality of split
  splitpoint, feature_index = split(x,y,minleaf,nfeat)

  #if the minleaf  constraint is not sasisfied, so if no split can be found that
  #creates a node with fewer than minleaf observations is not acceptable.
  #leaf node created
  if feature_index == None:
    return DecisionNode(value=y, leaf=True)

  #left
  y_l = y[x[:,feature_index]<=splitpoint]
  x_l = x[np.where(x[:,feature_index]<=splitpoint)]

  #right
  y_r = y[x[:,feature_index]>splitpoint]
  x_r = x[np.where(x[:,feature_index]>splitpoint)]

  #recursion
  left_tree = tree_grow(x_l,y_l, nmin, minleaf, nfeat)
  right_tree = tree_grow(x_r,y_r, nmin, minleaf, nfeat)

  return DecisionNode(feature=feature_index, splitpoint=splitpoint, left=left_tree, right=right_tree, value=y)


def tree_pred_one(x, tr):
    '''
    This function recursively predicts the y value for a single datapoint with features x, using a decision tree tr 
    using majority values in the leaf nodes.
    input:
    - x (1d np array (float)): the features for 1 x datapoint
    - tr(DecisionNode): the decision tree
    returns:
    - an int with a y label for x
    '''

    # if a leaf node is found
    if tr.is_leaf_node() == True:
        counter = Counter(tr.value)
        majority_value = counter.most_common(1)[0][0]
        return majority_value
    
    # otherwise see whether we should go left or right in the decision tree
    split_feature = tr.feature
    splitpoint = tr.splitpoint

    if x[split_feature] > splitpoint:
        return tree_pred_one(x,tr.right)
    if x[split_feature] <= splitpoint:
        return tree_pred_one(x,tr.left)


def tree_pred(x, tr):
    '''
    This function uses the tree_pred_one function to predict the y values for multiple datapoint with features x, 
    using a decision tree tr using majority values in the leaf nodes.
    input:
    - x (2d np array (float)): the features for multiple datapoint
    - tr(DecisionNode): the decision tree
    returns:
    - a 1d np array (int) with y labels for x
    '''
    # Tree prediction with just 1 tree, so without bagging
    predictions = []
    
    for datapoint in x:
        predictions.append(tree_pred_one(datapoint, tr))
    
    return np.array(predictions)


def tree_grow_b(x, y, nmin, minleaf, nfeat, m):
    '''
    A function that creates a decision tree using bagging.
    The recursive function uses the tree_grow function to build multiple decision trees.
    input:
    - x (2d np array (float)): the x values for all the features
    - y (2d np array (int)): the y values for all the features
    - nmin (int): the number of observations that a node must contain at least, for it to be allowed to be split
    - minleaf (int): the minimum number of observations required for a leaf node
    - nfeat (int): the number of features that should be considered for each split (chosen randomly)
    - m (int): denotes the number of bootstrap samples to be drawn and the number of decision trees that should be returned
    returns:
    - a list of DecisionNode (so decision tree) objects
    '''
    # Bagging tree grow, i.e. tree grow with bootstrap aggregating
    n = x.shape[0]
    tree_bags = []
    for i in range(m):
        index = np.random.choice(n, size=n, replace=True)
        new_x = x[index]

        tree = tree_grow(new_x,y,nmin,minleaf,nfeat)
        tree_bags.append(tree)

    return tree_bags # should be a list of trees


def tree_pred_b(x, tree_bags):
    '''
    A function that uses the tree_pred function to predict what each decision tree from multiple decision tree (bagging) would 
    predict the labels for multiple datapoint to be based on its features. The function then find the majority predicition for
    all the decision trees.
    input:
    - x (2d np array (float)): the features for multiple datapoint
    - treebags (list of DecisionNode objects): multiple decision trees
    returns:
    - the majority prediction labels for multiple x input datapoints
    '''
    # Tree prediction with bagging
    # for each store what the trees would predict
    predictions = []
    for tree in tree_bags:
        predictions.append(tree_pred(x, tree))

    #find the majority prediction
    Y = stats.mode(np.array(predictions).T, axis=1, keepdims=False)

    return Y.mode.flatten() # Should return a vector y where y[i] contains the predicted class label for row i of x