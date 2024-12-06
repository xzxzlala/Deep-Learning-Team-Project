# Required Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from imageio import imread
from skimage.transform import resize
import datetime
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, BatchNormalization, TimeDistributed, 
    Dense, Dropout, Flatten, SimpleRNN, Activation
)
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import warnings

# Suppress Warnings
warnings.filterwarnings("ignore")

# Seed Initialization
np.random.seed(42)
tf.random.set_seed(42)

# Paths and Dataset
train_file = '/kaggle/input/gesture-recognition-dataset/train.csv'
val_file = '/kaggle/input/gesture-recognition-dataset/val.csv'

train_data = pd.read_csv(train_file, dtype=object)
print(f"Training Data Shape: {train_data.shape}")
print(train_data.info())

# Parameters
num_frames, img_width, img_height = 30, 120, 120
num_classes, channels = 5, 3
batch_size = 16
train_files = np.random.permutation(open(train_file).readlines())
val_files = np.random.permutation(open(val_file).readlines())

# Data Generator Function
def data_generator(source_path, folder_list, batch_size):
    print(f"Source: {source_path}, Batch Size: {batch_size}")
    frame_indices = list(range(num_frames))
    while True:
        shuffled_folders = np.random.permutation(folder_list)
        batches = len(folder_list) // batch_size
        for i in range(batches):
            batch_images = np.zeros((batch_size, num_frames, img_width, img_height, channels))
            batch_labels = np.zeros((batch_size, num_classes))
            for j in range(batch_size):
                folder_path = os.path.join(source_path, shuffled_folders[j + i * batch_size].split(";")[0])
                images = os.listdir(folder_path)
                for k, idx in enumerate(frame_indices):
                    img = imread(os.path.join(folder_path, images[idx])).astype(np.float32)
                    img_resized = resize(img, (img_width, img_height)).mean(axis=-1, keepdims=True)
                    batch_images[j, k] = (img_resized / 127.5) - 1
                label = int(shuffled_folders[j + i * batch_size].split(";")[2])
                batch_labels[j, label] = 1
            yield batch_images, batch_labels

        # Handle Remaining Data
        remaining = len(folder_list) % batch_size
        if remaining:
            batch_images = np.zeros((remaining, num_frames, img_width, img_height, channels))
            batch_labels = np.zeros((remaining, num_classes))
            for j in range(remaining):
                folder_path = os.path.join(source_path, shuffled_folders[j + batches * batch_size].split(";")[0])
                images = os.listdir(folder_path)
                for k, idx in enumerate(frame_indices):
                    img = imread(os.path.join(folder_path, images[idx])).astype(np.float32)
                    img_resized = resize(img, (img_width, img_height)).mean(axis=-1, keepdims=True)
                    batch_images[j, k] = (img_resized / 127.5) - 1
                label = int(shuffled_folders[j + batches * batch_size].split(";")[2])
                batch_labels[j, label] = 1
            yield batch_images, batch_labels

# Model Definition
model = Sequential([
    TimeDistributed(Conv2D(16, (3, 3), activation='relu', padding='same'), input_shape=(num_frames, img_width, img_height, channels)),
    TimeDistributed(BatchNormalization()),
    TimeDistributed(MaxPooling2D((2, 2))),
    
    TimeDistributed(Conv2D(32, (3, 3), activation='relu', padding='same')),
    TimeDistributed(BatchNormalization()),
    TimeDistributed(MaxPooling2D((2, 2))),
    
    TimeDistributed(Conv2D(64, (3, 3), activation='relu', padding='same')),
    TimeDistributed(BatchNormalization()),
    TimeDistributed(MaxPooling2D((2, 2))),
    
    TimeDistributed(Flatten()),
    SimpleRNN(64),
    Dropout(0.25),
    Dense(64, activation='relu'),
    Dropout(0.25),
    Dense(num_classes, activation='softmax')
])

model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
print(model.summary())

# Callbacks
current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
model_dir = f"model_{current_time}"
os.makedirs(model_dir, exist_ok=True)

model_checkpoint = ModelCheckpoint(
    filepath=os.path.join(model_dir, "model-{epoch:02d}-{val_loss:.2f}.h5"),
    save_best_only=True, monitor='val_loss', verbose=1
)

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, verbose=1)
callbacks = [model_checkpoint, reduce_lr]

# Training
train_gen = data_generator('/kaggle/input/gesture-recognition-dataset/train', train_files, batch_size)
val_gen = data_generator('/kaggle/input/gesture-recognition-dataset/val', val_files, batch_size)

steps_per_epoch = len(train_files) // batch_size + (len(train_files) % batch_size > 0)
val_steps = len(val_files) // batch_size + (len(val_files) % batch_size > 0)

history = model.fit(
    train_gen, validation_data=val_gen,
    epochs=40, steps_per_epoch=steps_per_epoch,
    validation_steps=val_steps, callbacks=callbacks
)

# Plot Metrics
history_df = pd.DataFrame(history.history)

plt.figure(figsize=(12, 5))
plt.plot(history_df['loss'], label='Training Loss', color='darkred')
plt.plot(history_df['val_loss'], label='Validation Loss', color='blue')
plt.title('Model Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

plt.figure(figsize=(12, 5))
plt.plot(history_df['accuracy'], label='Training Accuracy', color='darkgreen')
plt.plot(history_df['val_accuracy'], label='Validation Accuracy', color='purple')
plt.title('Model Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
