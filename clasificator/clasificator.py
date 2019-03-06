##################### Obtain wine datase from .csv file

def read_csv(location, sep=';'):

  # File reader library
  import pandas as pd

  data = pd.read_csv(location, sep, usecols=[0,1,2,3,4,5,6,7,8,9,10])
  return data

##################### Split dataset into training and test datasets

def split_dataset(dataset):

    # Doesn't include the last column, which is the target
    features = dataset[:,:11]
    target = dataset[:,11]

    # Split dataset
    from sklearn.model_selection import train_test_split

    train_features, test_features, train_target, test_target = \
      train_test_split(features, target, test_size=0.30, random_state=7)

    return train_features, test_features, train_target, test_target

# If the target is > 5 it's a good wine. Bad wine in other case
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
  decision_tree = tree.DecisionTreeClassifier(max_depth=5)
  decision_tree.fit(train_features, train_target)

  return decision_tree

def fit_linear_regression(train_features, train_target):
  linear_regression = linear_model.LinearRegression()
  linear_regression.fit(train_features, train_target)
  
  return linear_regression

##################### Plotts and Graphs

import matplotlib.pyplot as plt
import seaborn as sns

def x_vs_y(x, x_label, y, y_label, title, save_path):
  plt.figure()
  plt.scatter(x, y, s=20, edgecolor="black", c="darkorange", label="data")
  plt.xlabel(x_label)
  plt.ylabel(y_label)
  plt.title(title)
  plt.savefig(save_path)

def correlation_matrix(frame, save_path):

  #Create Correlation df
  corr = frame.corr()
  print(corr.columns)
  #Plot figsize
  fig, ax = plt.subplots(figsize=(12, 14))
  #Generate Color Map
  colormap = sns.diverging_palette(220, 10, as_cmap=True)
  #Generate Heat Map, allow annotations and place floats in map
  sns.heatmap(corr, cmap=colormap, annot=True, fmt=".2f")
  #Apply xticks
  plt.xticks(range(len(corr.columns)), corr.columns);
  #Apply yticks
  plt.yticks(range(len(corr.columns)), corr.columns)
  plt.title('correlation matrix')
  #show plot
  #plt.show()

  #sns.heatmap(frame.corr(), annot=True, fmt=".2f")
  #plt.matshow(frame.corr())
  #plt.xticks(range(len(frame.columns)), frame.columns)
  #plt.yticks(range(len(frame.columns)), frame.columns)
  #plt.colorbar()
  plt.savefig(save_path)

##################### Main

import sys
import numpy as np

##### Obtain dataset and split into training and test

## White wine dataset
ww_dataset = read_csv(location = sys.argv[1])
#ww_feature_labels = list(ww_dataset)[:11]
#ww_target_label = list(ww_dataset)[11]
# Converts dataset into matrix
ww_dataset_values = ww_dataset.to_numpy()

## Split dataset into train and test
#train_features, test_features, train_target, test_target = \
#  split_dataset(ww_dataset_values)

# Used to train the classification models
#classification_train_target = transform_target_values(train_target)
#classification_test_target = transform_target_values(test_target)

#### Train models

## Classification models
correlation_matrix(ww_dataset, 'plots/corr.png')

## Classification models
  
  #x_vs_y(ww_dataset_values[:,i], ww_feature_labels[i], ww_dataset_values[:,11], ww_target_label, '%s vs quality' % ww_feature_labels[i], 'plots/scatters/%s.png' % i)
  
  #logistic_regression = fit_logistic_regression(np.reshape(train_features[:,i], (-1, 1)),
   #                                             classification_train_target)
  #decision_tree = fit_decision_tree(np.reshape(train_features[:,i], (-1, 1)),
    #                                classification_train_target)
  
  ## Models accuracy
  #print('Logistic regression accuracy: ')
  #print(logistic_regression.score(np.reshape(test_features[:,i], (-1, 1)),
     #                             classification_test_target))
  #print('Decision tree accuracy: ')
  #print(decision_tree.score(np.reshape(test_features[:,i], (-1, 1)),
      #                      classification_test_target))


  
  #dot_data = tree.export_graphviz(decision_tree, out_file='plots/%s.gv' % i,
#                                  feature_names=ww_feature_labels[i:i+1],
 #                                 class_names=ww_target_label,
  #                                filled=True, rounded=True,
  #                                special_characters=True)


# logistic_regression = fit_logistic_regression(train_features[:, [5,6]],
#                                               classification_train_target)
  
# decision_tree = fit_decision_tree(train_features[:, [5,6]],
#                                   classification_train_target)
  
# ## Models accuracy
# print('Logistic regression accuracy: ')
# print(logistic_regression.score(test_features[:, [5,6]],
#                                 classification_test_target))
# print('Decision tree accuracy: ')
# print(decision_tree.score(test_features[:, [5,6]],
#                           classification_test_target))

  
## Linear regression

# linear_regression = fit_linear_regression(train_features, train_target)

# from sklearn.feature_selection import chi2

# scores, pvalues = chi2(ww_dataset_values[:,:11], ww_dataset_values[:,11])
                           
# print('Linear regression accuracy: ')
# print(linear_regression.score(test_features, test_target))

# print('P-values: ')
# print(ww_feature_labels)
# print(pvalues)

## Plotts

# DOT: Graph description language
# dot_data = tree.export_graphviz(decision_tree, out_file='plots/5and6.gv',
#                                 feature_names=ww_feature_labels[5:7],
#                                 class_names=ww_target_label,
#                                 filled=True, rounded=True,
#                                 special_characters=True)


# good = bad = 0
# for i in range(len(ww_dataset_values[:,11])):

#   if ww_dataset_values[i, 11] > 5:
#     good += 1
#   else:
#     bad += 1

# print('good ')
# print(good)
# print('bad ')
# print(bad)
# print(good + bad)

