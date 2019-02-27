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
    features = matrix[:][:12]
    target = matrix[:][11]

    # Split dataset
    from sklearn.model_selection import train_test_split

    train_features, test_features, train_target, test_target = ()
    
    return 0

##################### Main

##### Obtain dataset and split into training and test

# White wine dataset
ww_dataset = read_csv(location = '../whitewine.csv')

m = ww_dataset.to_numpy()
print(m[0][:1])
