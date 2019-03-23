####
# X vs Y scatter plot
####
def x_vs_y(x, x_label, y, y_label, title, save_path):
  plt.figure()
  plt.scatter(x, y, s=20, edgecolor="black", c="darkorange", label="data")
  plt.xlabel(x_label)
  plt.ylabel(y_label)
  plt.title(title)
  plt.savefig(save_path)

####
# Correlation matrix plot
####
def correlation_matrix(frame, save_path):

  #Create Correlation df
  corr = frame.corr()
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

  plt.savefig(save_path)
