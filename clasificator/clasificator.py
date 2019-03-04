##################### Obtain wine datase from .csv file

def read_csv(location, sep=';'):

  # File reader library
  import pandas as pd

  data = pd.read_csv(location, sep)
  return data

##################### Split dataset into training and test datasets

def split_dataset(dataset):

    # Convert the dataset into a matrix
    matrix = dataset.to_numpy()
    # Doesn't include the last column, which is the target
    features = matrix[:,:11]
    target = matrix[:,11]

    # Classify into Good and Bad wines
    for i in range(len(target)):

      if target[i] > 5: # Good wine
        target[i] = 1
      else: # Bad wine
        target[i] = 0

    # Split dataset
    from sklearn.model_selection import train_test_split

    train_features, test_features, train_target, test_target = \
      train_test_split(features, target, test_size=0.30)

    return train_features, test_features, train_target, test_target

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

##################### Main

import sys

##### Obtain dataset and split into training and test

## White wine dataset
ww_dataset = read_csv(location = sys.argv[1])

## Split dataset into train and test
train_features, test_features, train_target, test_target = \
  split_dataset(ww_dataset)

#### Train models

for i in range(11):
  print(i)
  logistic_regression = fit_logistic_regression(train_features[:][i],
                                                train_target)
  decision_tree = fit_decision_tree(train_features[:][i], train_target)

  ## Models accuracy
  print('Logistic regression accuracy: ')
  print(logistic_regression.score(test_features, test_target))
  print('Decision tree accuracy: ')
  print(decision_tree.score(test_features, test_target))

## Plotts

# DOT: Graph description language
# dot_data = tree.export_graphviz(decision_tree, out_file='plots/test.gv',
#                                 feature_names=list(ww_dataset)[:11],
#                                 class_names=list(ww_dataset)[11],
#                                 filled=True, rounded=True,
#                                 special_characters=True)

