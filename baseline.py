import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import label_binarize

data_path = "./gesture-recognition-dataset/"


# Define data paths
train_csv_path = os.path.join(data_path, "train.csv")
val_csv_path = os.path.join(data_path, "val.csv")
train_data_path = os.path.join(data_path, "train")
val_data_path = os.path.join(data_path, "val")

# Step 1: Load the data
train_df = pd.read_csv(train_csv_path, sep=";", header=None, names=['subfolder', 'gesture', 'label'])
val_df = pd.read_csv(val_csv_path, sep=";", header=None, names=['subfolder', 'gesture', 'label'])

# Extract labels
train_labels = train_df['label'].values
val_labels = val_df['label'].values

# Step 2: ACC Baseline (Stratified Guess)
dummy_clf_acc = DummyClassifier(strategy="stratified", random_state=42)
dummy_clf_acc.fit(np.zeros((len(train_labels), 1)), train_labels)  # Dummy input, only labels are used

# Predict on validation set
val_pred_acc = dummy_clf_acc.predict(np.zeros((len(val_labels), 1)))

# Compute ACC
acc_baseline = accuracy_score(val_labels, val_pred_acc)
print("ACC Baseline (Stratified Guess):", acc_baseline)

# Step 3: AUC Baseline (Stratified Guess)
# One-vs-Rest: Binarize labels for multi-class AUC
unique_classes = np.unique(train_labels)  # 获取所有类别
train_labels_bin = label_binarize(train_labels, classes=unique_classes)
val_labels_bin = label_binarize(val_labels, classes=unique_classes)

# DummyClassifier with stratified strategy
dummy_clf_auc = DummyClassifier(strategy="stratified", random_state=42)
dummy_clf_auc.fit(np.zeros((len(train_labels), 1)), train_labels)

# Predict probabilities for AUC computation
val_pred_proba = dummy_clf_auc.predict_proba(np.zeros((len(val_labels), 1)))

# Compute macro-average AUC
auc_baseline = roc_auc_score(val_labels_bin, val_pred_proba, average="macro", multi_class="ovr")
print("AUC Baseline (Stratified Guess):", auc_baseline)