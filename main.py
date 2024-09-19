import numpy as np
import collections
import pandas as pd

def impurity(a):
  #Gini index: i(t) = p(0|t)p(1|t) = p(0|t)(1 − p(0|t))
  i = (np.count_nonzero(a == 0)/len(a))*(1-np.count_nonzero(a == 0)/len(a)) 
  return i

def bestsplit(x,y):
  np.sort(np.unique(x))

df = pd.read_csv('credit.txt').to_numpy()
df2 = pd.read_csv('pima indians.txt', header=None).to_numpy()

x = df2[:, 0:8]
y = df2[:, 8]


def bestsplit(x,y):
  impurity_parent = impurity(y)
  
  x_sorted = np.sort(np.unique(x))
  x_splitpoints = (x_sorted[0:(len(x_sorted)-1)]+x_sorted[1:(len(x_sorted))])/2
  
  reductions = []
  for s in x_splitpoints:
    left = impurity(y[x <= s])
    l = len(y[x <= s])
    right = impurity(y[x > s])
    r = len(y[x > s])
    reduction = impurity_parent - ((l*left) + r*(right))
    reductions.append(reduction)

  best_splitpoint = x_splitpoints[reductions.index(max(reductions))]

  return best_splitpoint
  


# x == data matrix (2-dimens.) contain the attribute values
# y is the vector (1- dimens. array) of class labels. Class labels are 0 or 1.

def tree_grow(x, y, nmin, minleaf, nfeat):
    
    tree = 

    return tree

def tree_pred(x, tr=tree):
    
    
    return y # Output pred. (1-dimens.)



def main():
  tree = tree_grow()
  tree_pred()

main()
