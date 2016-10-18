import sys
import math
import numpy as np
import random
import os

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
      self.attributes = None
      
	
   def train(self, X_train, Y_train, attributes) :
      # receives a list of objects of type Example
      self.label = pluralityValue(Y_train) # Set label to the plurality value
      self.attributes = attributes
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
      elif  len(self.attributes) == 0 or self.curr_depth == self.depth_limit:
         self.leaf = pluralityValue(Y_train)
         return self
      else:
         # Check entropy of all attributes
         # We don't need to remove the attributes already used
         # Using an attribute already used will result in 0 information gain
         if self.split_random == 0:
            self.A = argMax(set(self.attributes),X_train,Y_train)
         else:
            self.A = random.sample(self.attributes,1)[0]
         self.attributes.remove(self.A)
         for v_k in range(2):
            X_exs = [e for e in X_train if e[self.A] == v_k]
            Y_exs = [Y_train[i] for i in range(len(Y_train)) if X_train[i][self.A] == v_k]
            subtree = DecisionTree(self.split_random,self.depth_limit,self.curr_depth+1,v_k)
            self.branch.append(subtree.train(X_exs,Y_exs,set(self.attributes)))
         return self
      None
         
   def predict(self, X_train):
      # receives a list of booleans
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
   # It's OK to pass in an empty list here
   # It will just return 0 and wont contribute to infoGain
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
   maxIndex = {-1}
   max = 0
   for a in attributes:
      infoGain = entropy(y) - np.sum([float(len(s_v(x,y,v,a))) / float(len(y)) * entropy(s_v(x,y,v,a)) for v in range(2)])
      if infoGain > max:
         maxIndex = {a}
         max = infoGain
      elif infoGain == max:
         maxIndex.add(a)
   indexOut = random.sample(maxIndex,1)[0] # If more than one max, set to random
   if indexOut == -1:
      indexOut = random.sample(attributes,1)[0]
   return indexOut

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

# Read command line arguments
train_file = sys.argv[1]
tree_type = sys.argv[2]
depth = int(sys.argv[3])
test_file = sys.argv[4]
output_file = sys.argv[5]

# Read train and test file
X_train, Y_train = read_datafile(train_file)
X_test, Y_test = read_datafile(test_file)

# Create and train a decision tree
my_attributes = {i for i in range(len(X_train[0]))}   # Set of attributes
splitFlag = 0 if tree_type == 'I' else 1  # INFO GAIN = 0, RANDOM = 1
dt = DecisionTree(splitFlag,depth,0,1)
dt.train(X_train,Y_train,set(my_attributes))

print "compute_accuracy = " + str(compute_accuracy(dt,X_test,Y_test))

# Print to output file
if not os.path.exists(output_file):
   file(output_file, 'w').close()
f = open(output_file,'a+')
f.write(str(compute_accuracy(dt,X_test,Y_test)) + "\n")
f.close()