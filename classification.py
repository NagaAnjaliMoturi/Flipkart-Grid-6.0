import numpy as np
import pandas as pd
import os
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
from random import shuffle
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, BatchNormalization, SeparableConv2D, MaxPooling2D, Dropout, Flatten, Dense
import keras

# Set directory paths
dir_path = 'dataset/Train'
dir_path_test = 'dataset/Test'

# Load and preprocess images
def load_rand():
    X = []
    for sub_dir in tqdm(os.listdir(dir_path)):
        path_main = os.path.join(dir_path, sub_dir)
        i = 0
        for img_name in os.listdir(path_main):
            if i >= 6:
                break
            img = cv2.imread(os.path.join(path_main, img_name))
            img = cv2.resize(img, (100, 100))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            X.append(img)
            i += 1
    return np.array(X)

X = load_rand()

def load_rottenvsfresh():
    quality = ['fresh', 'rotten']
    X, Y, z = [], [], []
    for cata in tqdm(os.listdir(dir_path)):
        path_main = os.path.join(dir_path, cata)
        label = 0 if quality[0] in cata else 1
        for img_name in os.listdir(path_main):
            img = cv2.imread(os.path.join(path_main, img_name))
            img = cv2.resize(img, (100, 100))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            z.append([img, label])
    shuffle(z)
    for images, labels in tqdm(z):
        X.append(images)
        Y.append(labels)
    return np.array(X), np.array(Y)

X, Y = load_rottenvsfresh()

def load_rottenvsfresh_valset():
    quality = ['fresh', 'rotten']
    X, Y, z = [], [], []
    for cata in tqdm(os.listdir(dir_path_test)):
        path_main = os.path.join(dir_path_test, cata)
        label = 0 if quality[0] in cata else 1
        for img_name in os.listdir(path_main):
            img = cv2.imread(os.path.join(path_main, img_name))
            img = cv2.resize(img, (100, 100))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            z.append([img, label])
    shuffle(z)
    for images, labels in tqdm(z):
        X.append(images)
        Y.append(labels)
    return np.array(X), np.array(Y)

X_val, Y_val = load_rottenvsfresh_valset()

# Normalize the images
X = X / 255.0
X_val = X_val / 255.0

# Build the model
mobilenetv2_model = tf.keras.applications.MobileNetV2(weights='imagenet', include_top=False, input_shape=(100, 100, 3))

for layer in mobilenetv2_model.layers:
    layer.trainable = False

model = Sequential()
model.add(mobilenetv2_model)
model.add(BatchNormalization())
model.add(SeparableConv2D(64, (3, 3), padding='same', activation='relu'))
model.add(SeparableConv2D(64, (3, 3), padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.4))
model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

# Callbacks
lr_rate = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=6, verbose=1, mode='max', min_lr=0.00002, cooldown=2)
check_point = tf.keras.callbacks.ModelCheckpoint(filepath='modelcheckpt.keras', monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=False, mode='min')

# Train the model
history = model.fit(X, Y, batch_size=2, validation_data=(X_val, Y_val), epochs=10, callbacks=[check_point])

# Plot training history
plt.figure(figsize=(20, 12))
plt.subplot(1, 2, 1)
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.grid(True)
plt.legend()

plt.subplot(1, 2, 2)
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.plot(history.history["accuracy"], label="Training Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.grid(True)
plt.legend()
plt.show()

# Evaluate the model
model.evaluate(X_val, Y_val)

# Save the model
model.save('rottenvsfresh.keras')

# Load and evaluate the model
new_model = tf.keras.models.load_model('rottenvsfresh.keras')
new_model.evaluate(X_val, Y_val)
