################################################################################
# Author: Hamilton Tobon Mosquera
# Date: 25/03/2019
################################################################################

import sys
import pickle
import pandas as pd

if len(sys.argv) != 4:
    print('Usage: \n',
          '$ python final_model.py linear-regression.model decision-tree.model <input-data>.csv\n',
          'Where <input-data> is your .csv file with your input data')
    exit(1)

# Read dataset
print('Reading dataset separated by ;')

dataset = pd.read_csv(sys.argv[3], ';')
values = dataset.to_numpy()
columns = [1,2,5,7,8,9,10] # Features used
features = values[:, columns]

# Load models
linear_regression = pickle.load(open(sys.argv[1], 'rb'))
decision_tree = pickle.load(open(sys.argv[2], 'rb'))

# Make predictions
out_file = open('prediction-output.csv', 'w+')

for row in features:
    row = row.reshape(1, -1)
    good = decision_tree.predict(row)
    if good == 1.0:

        out_file.write('%d, %s\n' % (linear_regression.predict(row),
                                     'BUENO'))
    else:
        out_file.write('%d, %s\n' % (linear_regression.predict(row),
                                     'MALO'))

print('Prediction done!')
# Close output file
out_file.close()
