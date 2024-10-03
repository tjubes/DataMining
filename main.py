import numpy as np
import collections
import pandas as pd
import random

class DecisionNode():
    def __init__(self, feature=None, splitpoint=None, left=None, right=None, value=None, leaf=False):
        self.feature = feature          # Feature to split on, column index
        self.splitpoint = splitpoint    # Threshold value for the split
        self.left = left                # Left subtree
        self.right = right              # Right subtree
        self.value = value              # If leaf node, the predicted class
        self.leaf = leaf                # If node is leaf node

    def is_leaf_node(self):
        return self.leaf is True

def impurity(a):
  i = (np.count_nonzero(a == 0)/len(a))*(1-np.count_nonzero(a == 0)/len(a))
  return i

def bestsplit(x,y, minleaf):
# this function is for a single feature
#return best splitpoint, reduction of splitpoint (gini index)

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
  #this function finds which feature has the best split, using the best_split function
  # which feature do we split on, and what is the threshold?
  best_reduction = 0
  best_splitpoint = None
  column_i = None
  i = 0
  n_features = len(x.T)

  #which random features we split on
  random_features = random.sample(range(0, n_features), nfeat)

  for index in random_features:
    splitpoint, reduction = bestsplit(x[:, index],y,minleaf)

    if reduction > best_reduction:
      best_reduction = reduction
      best_splitpoint = splitpoint
      column_i = i
    i+=1
  return best_splitpoint, column_i

def tree_grow(x, y, nmin, minleaf, nfeat):
  observations = len(x)

  #the number of observations that a node must contain at least, for it to be
  #allowed to be split
  #also if the node is already pure, splitting is not needed anymore
  if observations<nmin or np.all(y == 0): #is np.all(y == 0) correct?
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

df = pd.read_csv('pima.txt', header=None).to_numpy()

x = df[:, 0:8]
y = df[:, 8]
print(split(x,y,5,3))
print(tree_grow(x,y,10,10,5).right.value)