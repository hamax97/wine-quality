####
# IMPORTS

# File reader library
import pandas as pd
import matplotlib.pyplot as plt
import sys
import numpy as np
from sklearn import linear_model
from sklearn import tree
#import seaborn as sns

import plots
import models
import utilities

####

################################################################################
# MAIN
################################################################################

#####
# Obtain dataset and split into training and test
####

# White wine dataset
ww_dataset = pd.read_csv(sys.argv[1], ';')
ww_feature_labels = ww_dataset.columns[:11]
ww_target_label = ww_dataset.columns[11]

# Clean dataset
#utilities.missing_values(ww_dataset)
print(ww_dataset.dtypes)

# Converts dataset into numpy matrix
# ww_dataset_values = ww_dataset.to_numpy()

# Split dataset into train and test
# train_features, test_features, train_target, test_target = \
#   split_dataset(ww_dataset_values)

# Used to train the classification models
# classification_train_target = transform_target_values(train_target)
# classification_test_target = transform_target_values(test_target)

####
# Train classification models
####

# Best combination
# columns = [1, 2, 5, 6, 7, 10]

# logistic_regression = fit_logistic_regression(train_features[:, columns],
#                                               classification_train_target)
# decision_tree = fit_decision_tree(train_features[:, columns],
#                                   classification_train_target)

# ## Models accuracy
# print('Logistic regression accuracy: ')
# print(logistic_regression.score(test_features[:, columns],
#                                 classification_test_target))
# print('Decision tree accuracy: ')
# print(decision_tree.score(test_features[:, columns],
#                           classification_test_target))

####
# Train Linear regression
####
# linear_regression = fit_linear_regression(train_features[:, columns], train_target)

# print('Linear regression accuracy: ')
# print(linear_regression.score(test_features[:, columns], test_target))

