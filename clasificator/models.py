#####
# Logistic Regression fitter
#####
def fit_logistic_regression(train_features, train_target):
  logistic_regression = \
    linear_model.LogisticRegression(solver='lbfgs', multi_class='multinomial')

  logistic_regression.fit(train_features, train_target)
  return logistic_regression

#####
# Decision Tree fitter
#####
def fit_decision_tree(train_features, train_target):
  decision_tree = tree.DecisionTreeClassifier()
  decision_tree.fit(train_features, train_target)

  return decision_tree

#####
# Linear Regression fitter
#####
def fit_linear_regression(train_features, train_target):
  linear_regression = linear_model.LinearRegression()
  linear_regression.fit(train_features, train_target)

  return linear_regression
