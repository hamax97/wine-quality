##################### Obtain wine datase from .csv file

def read_csv(location, sep=';'):

  # File reader library
  import pandas as pd

  data = pd.read_csv(location, sep)
  return data

##################### Split dataset into training and test datasets

def split_dataset(dataset):

    # Doesn't include the last column, which is the target
    features = dataset[:,:11]
    target = dataset[:,11]

    # Split dataset
    from sklearn.model_selection import train_test_split

    train_features, test_features, train_target, test_target = \
      train_test_split(features, target, test_size=0.30)

    return train_features, test_features, train_target, test_target

def transform_target_values(target):

  transformed_target = target.copy()
  
  # Classify into Good and Bad wines
  for i in range(len(target)):
    
    if transformed_target[i] > 5: # Good wine
      transformed_target[i] = 1
    else: # Bad wine
      transformed_target[i] = 0

  return transformed_target

##################### Fit models

from sklearn import linear_model
from sklearn import tree

def fit_logistic_regression(train_features, train_target):
  logistic_regression = \
    linear_model.LogisticRegression(solver='lbfgs', multi_class='multinomial')

  logistic_regression.fit(train_features, train_target)
  return logistic_regression

def fit_decision_tree(train_features, train_target):
  decision_tree = tree.DecisionTreeClassifier()
  decision_tree.fit(train_features, train_target)

  return decision_tree

def fit_linear_regression(train_features, train_target):
  linear_regression = linear_model.LinearRegression()
  linear_regression.fit(train_features, train_target)
  
  return linear_regression

##################### Main

import sys
import numpy as np

##### Obtain dataset and split into training and test

## White wine dataset
ww_dataset = read_csv(location = sys.argv[1])
ww_dataset_labels = list(ww_dataset)[:11]
# Converts dataset into matrix
ww_dataset_values = ww_dataset.to_numpy()

## Split dataset into train and test
train_features, test_features, train_target, test_target = \
  split_dataset(ww_dataset_values)

# Used to train the classification models
classification_train_target = transform_target_values(train_target)
classification_test_target = transform_target_values(test_target)

#### Train models

## Classification models
for i in range(11):
  print(i)
  logistic_regression = fit_logistic_regression(np.reshape(train_features[:,i], (-1, 1)),
                                                classification_train_target)
  decision_tree = fit_decision_tree(np.reshape(train_features[:,i], (-1, 1)),
                                    classification_train_target)

  ## Models accuracy
  print('Logistic regression accuracy: ')
  print(logistic_regression.score(np.reshape(test_features[:,i], (-1, 1)),
                                  classification_test_target))
  print('Decision tree accuracy: ')
  print(decision_tree.score(np.reshape(test_features[:,i], (-1, 1)),
                            classification_test_target))

  
linear_regression = fit_linear_regression(train_features, train_target)

## Linear regression

from sklearn.feature_selection import chi2

scores, pvalues = chi2(ww_dataset_values[:,:11], ww_dataset_values[:,11])
                           
print('Linear regression accuracy: ')
print(linear_regression.score(test_features, test_target))

print('P-values: ')
print(ww_dataset_labels)
print(pvalues)

## Plotts

# DOT: Graph description language
# dot_data = tree.export_graphviz(decision_tree, out_file='plots/test.gv',
#                                 feature_names=ww_dataset_labels,
#                                 class_names=list(ww_dataset)[11],
#                                 filled=True, rounded=True,
#                                 special_characters=True)

