import math
import numpy as np
import random

def read_datafile(fname, attribute_data_type = 'integer'):
   inf = open(fname,'r')
   lines = inf.readlines()
   inf.close()
   #--
   X = []
   Y = []
   for l in lines:
      ss=l.strip().split(',')
      temp = []
      for s in ss:
         if attribute_data_type == 'integer':
            temp.append(int(s))
         elif attribute_data_type == 'string':
            temp.append(s)
         else:
            print("Unknown data type");
            exit();
      X.append(temp[:-1])
      Y.append(int(temp[-1]))
   return X, Y

#===
class DecisionTree :
   def __init__(self, split_random, depth_limit, curr_depth = 0, default_label = 1):
      self.split_random = split_random # if True splits randomly, otherwise splits based on information gain 
      self.depth_limit = depth_limit
      self.curr_depth = curr_depth
      self.A = None
      self.branch = []
      self.label = default_label
      self.leaf = None
      
	
   def train(self, X_train, Y_train) :
      # receives a list of objects of type Example
      # TODO: implement decision tree training
      # Set label to the plurality value
      self.label = pluralityValue(Y_train)
      # Check i X_train is empty
      if len(X_train) is 0:
         # When we are searching in testing, if we get back none, should do plurality value
         return None
      elif all(y==Y_train[0] for y in Y_train):
         # TODO: return the classification
         # should still return a tree here
         # just need to acknowledge it is a leaf
         # could look at the branch to see if it is empty or check if label has a value
         self.leaf = Y_train[0]
         return self
      elif self.curr_depth == self.depth_limit:
         self.leaf = pluralityValue(Y_train)
         return self
      else:
         # Check entropy of all attributes
         # We don't need to remove the attributes already used
         # Using an attribute already used will result in 0 information gain
         self.A = argMax(len(X_train[0]),X_train,Y_train)
         for v_k in range(2):
            X_exs = [e for e in X_train if e[self.A] == v_k]
            #print X_exs
            Y_exs = [Y_train[i] for i in range(len(Y_train)) if X_train[i][self.A] == v_k]
            #print Y_exs
            subtree = DecisionTree(self.split_random,self.depth_limit,self.curr_depth+1,v_k)
            self.branch.append(subtree.train(X_exs,Y_exs))
         return self
      None
         
   def predict(self, X_train):
      # receives a list of booleans
      # TODO: implement decision tree prediction
      if self.leaf is not None:
         return self.leaf
      elif self.branch[X_train[self.A]] is not None:
         return self.branch[X_train[self.A]].predict(X_train)
      else:
         return self.label
      None

#===
# Only works for attributes having value 0 or 1
def entropy(y):
   entropy = 0.0
   if y.count(0) is not 0 and y.count(1) is not 0:
      for i in range(2) : entropy += -(y.count(i)/float(len(y)))*math.log((y.count(i)/float(len(y))),2)
   return entropy

#===
def importance(x,y):
   None

#===
# Returns a subset for a given attribute and attribute value
def s_v(x,y,v,a):
   return [y[i] for i in range(len(y)) if x[i][a] == v]

def argMax(attributes,x,y):
   maxIndex = -1
   max = 0
   for a in range(attributes):
      #print s_v(x,y,0,0)
      #infoGain = entropy(y) - np.sum([0,1,1])
      infoGain = entropy(y) - np.sum([float(len(s_v(x,y,v,a))) / float(len(y)) * entropy(s_v(x,y,v,a)) for v in range(2)])
      if infoGain > max:
         #print infoGain
         maxIndex = a
   #print maxIndex
   return maxIndex

#===
# Returns the most common output value among a set of examples
# breaks ties evenly
def pluralityValue(y):
   if y.count(0) > y.count(1):
      return 0
   elif y.count(1) > y.count(0):
      return 1
   else:
      return random.randint(0, 1)

#===
def compute_accuracy(dt_classifier, X_test, Y_test):
   numRight = 0
   for i in range(len(Y_test)):
      x = X_test[i]
      y = Y_test[i]
      if y == dt_classifier.predict(x) :
         numRight += 1
   return (numRight*1.0)/len(Y_test)

#==============================================
#==============================================
X_train, Y_train = read_datafile('train.txt')
X_test, Y_test = read_datafile('test.txt')
# TODO: write your code

#x = [[0,0,0],[1,1,1],[0,1,0],[0,0,1]]
#y = [0,1,1,0]
#a = 0

#argMax(3,x,y)

depthLimit = 4
splitFlag = 0
dt = DecisionTree(splitFlag,depthLimit,0,1)
#dt.train(x,y)
dt.train(X_train,Y_train)
#print dt.predict([1,0,0])
print compute_accuracy(dt,X_test,Y_test)

