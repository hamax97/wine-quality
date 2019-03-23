#####
# Split dataset into training and test datasets
#####
def split_dataset(dataset):

    # Doesn't include the last column, which is the target
    features = dataset[:,:11]
    target = dataset[:,11]

    # Split dataset
    from sklearn.model_selection import train_test_split

    train_features, test_features, train_target, test_target = \
      train_test_split(features, target, test_size=0.30, random_state=7)

    return train_features, test_features, train_target, test_target

#####
# If the target is > 5 it's a good wine. Bad wine in other case
#####
def transform_target_values(target):

  transformed_target = target.copy()

  # Classify into Good and Bad wines
  for i in range(len(target)):
    if transformed_target[i] > 5: # Good wine
      transformed_target[i] = 1
    else: # Bad wine
      transformed_target[i] = 0

  return transformed_target

####
# Print missing values count per feature
####
def missing_values(data_frame):

    missing_data = data_frame.isnull()
    for column in missing_data.columns.values.tolist():
        print(column)
        print(missing_data[column].value_counts())
        print('')
