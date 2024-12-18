# draw the data distribution of the dataset
import matplotlib.pyplot as plt
import os
import pandas as pd

data_path = "./gesture-recognition-dataset/"


# Define data paths
train_csv_path = os.path.join(data_path, "train.csv")
val_csv_path = os.path.join(data_path, "val.csv")
train_data_path = os.path.join(data_path, "train")
val_data_path = os.path.join(data_path, "val")

# Load the CSV files
train_df = pd.read_csv(train_csv_path)
val_df = pd.read_csv(val_csv_path)



train_label_counts = pd.read_csv(train_csv_path,sep=";", header=None, names=['subfolder', 'gesture', 'label'])['label'].value_counts()
val_label_counts = pd.read_csv(val_csv_path,sep=";", header=None, names=['subfolder', 'gesture', 'label'])['label'].value_counts()

label_to_gesture = {
    0: 'Left Swipe',
    1: 'Right Swipe',
    2: 'Stop',
    3: 'Thumbs Down',
    4: 'Thumbs Up'
}

train_labels = [label_to_gesture[label] for label in train_label_counts.index]
val_labels = [label_to_gesture[label] for label in val_label_counts.index]

plt.rcParams.update({'font.size': 14, 'font.weight': 'bold'})


plt.figure(figsize=(10, 10))
plt.pie(train_label_counts, labels=train_labels, autopct='%1.1f%%', textprops={'fontsize': 14, 'fontweight': 'bold'})
plt.title("Train Data Distribution", fontsize=16, fontweight='bold')
plt.savefig('train_data_distribution.png')
plt.show()


plt.figure(figsize=(10, 10))
plt.pie(val_label_counts, labels=val_labels, autopct='%1.1f%%', textprops={'fontsize': 14, 'fontweight': 'bold'})
plt.title("Test Data Distribution", fontsize=16, fontweight='bold')
plt.savefig('test_data_distribution.png')
plt.show()
