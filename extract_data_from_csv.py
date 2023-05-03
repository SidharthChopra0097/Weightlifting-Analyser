import pandas as pd
import numpy as np

#from google.colab import drive
#drive.mount('/content/drive')

# Read in the data from the CSV file
#right_bicep_angles_data = pd.read_csv('/content/drive/MyDrive/Bicep_right_angles.csv')
right_bicep_angles_data = pd.read_csv('dataset/Bicep_right_angles.csv')

# Extract the body angles and labels from the data
angles = right_bicep_angles_data.iloc[:, 0].values.reshape(-1, 1)
labels = np.ones(len(angles)) # Set all labels to 1 since the angles are correct


# Split the data into training, validation, and test sets
train_frac, val_frac, test_frac = 0.7, 0.2, 0.1
num_train = int(train_frac * len(right_bicep_angles_data))
num_val = int(val_frac * len(right_bicep_angles_data))
num_test = len(right_bicep_angles_data) - num_train - num_val

train_angles, train_labels = angles[:num_train], labels[:num_train]
val_angles, val_labels = angles[num_train:num_train+num_val], labels[num_train:num_train+num_val]
test_angles, test_labels = angles[num_train+num_val:], labels[num_train+num_val:]
