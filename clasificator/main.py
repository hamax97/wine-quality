import pandas as pd
import pickle
from sklearn import tree, linear_model
from utilities import transform_target_values

# Read data set
df = pd.read_csv('../dataset/whitewine.csv', ';')
labels = df.columns
data = df.to_numpy()

# Features used
columns = [1,2,5,7,8,9,10]
features = data[:, columns]
target = transform_target_values(data[:,11])

lr = linear_model.LinearRegression().fit(features, data[:, 11])
dt = tree.DecisionTreeClassifier().fit(features, target)

# Store models
pickle.dump(lr, open('../model/linear-regression.model', 'wb'))
pickle.dump(dt, open('../model/decision-tree.model', 'wb'))
