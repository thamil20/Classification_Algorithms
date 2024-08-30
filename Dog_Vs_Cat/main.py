# This program was made while following a tutorial that used a different dataset.
import tensorflow as tf
import os
import numpy as np
from matplotlib import pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from tensorflow.keras.metrics import Precision, Recall, BinaryAccuracy
import cv2

# Limit GPUs memory growth, avoiding OOM errors.
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

data_dir = "Dog_Vs_Cat\\dataset"

########################################################################

# Loading Data into TensorFlow
data = tf.keras.utils.image_dataset_from_directory(data_dir, batch_size=8)
data_iterator = data.as_numpy_iterator()

# Get another batch from the iterator
batch = data_iterator.next()

print(batch[0].shape)

# 0 = Cats, 1 = Dogs
print(batch[1])


def plotting(batch):
    fig, ax, = plt.subplots(ncols=4, figsize=(20,20))
    for idx, img in enumerate(batch[0][:4]):
        ax[idx].imshow(img)
        ax[idx].title.set_text(batch[1][idx])
    plt.show()

########################################################################

# Data Preprocessing
data = data.map(lambda x,y: (x/255, y))
scaled_iterator = data.as_numpy_iterator().next()

plotting(scaled_iterator)

# Splitting Data
# 70% of data in training
training_size = int(len(data)*.7)
# 20% of data in validating
validating_size = int(len(data)*.2)
# 10% of data in testing
testing_size = int(len(data)*.1)

# Check how much data was split. 
print(testing_size, validating_size, training_size)\

# In this case, 8 pictures were not split amongst the dataset. 
# There are 992 pictures in this dataset.


# Take or skip all data within the data set.
training = data.take(training_size)
validating = data.skip(training_size).take(validating_size)
testing = data.skip(training_size+validating_size).take(testing_size)

# Model Setup

model = Sequential()


# Model Initialization
model.add(Conv2D(16, (3,3), 1, activation='relu', input_shape=(256,256,3)))
model.add(MaxPooling2D())

model.add(Conv2D(32, (3,3), 1, activation='relu'))
model.add(MaxPooling2D())

model.add(Conv2D(16, (3,3), 1, activation='relu'))
model.add(MaxPooling2D())

model.add(Flatten())

model.add(Dense(256, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Model Compilation
model.compile('adam', loss=tf.losses.BinaryCrossentropy(), metrics= ['accuracy'])
print(model.summary())

# Training the Dataset
logdir = 'Dog_Vs_Cat\\logs'
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)

hist = model.fit(training, epochs=20, validation_data=validating, callbacks=[tensorboard_callback])

# Plotting Performance
# plot_loss is a function that is used to plot the loss of the model.
def plot_loss(hist):
    fig = plt.figure()
    plt.plot(hist.history['loss'], color='red', label='loss')
    plt.plot(hist.history['val_loss'], color='orange', label='val_loss')
    fig.suptitle('Loss', fontsize=20)
    plt.legend(loc="upper left")
    plt.show()


# plot_accuracy is a function that is used to plot the accuracy of the model.
def plot_accuracy(hist):
    fig = plt.figure()
    plt.plot(hist.history['accuracy'], color='green', label='accuracy')
    plt.plot(hist.history['val_accuracy'], color='teal', label='val_accuracy')
    fig.suptitle('Accuracy', fontsize=20)
    plt.legend(loc="upper left")
    plt.show()

# Show the plots of loss and accuracy    
plot_loss(hist=hist)
plot_accuracy(hist=hist)

# Evaluating Model Performance
precision = Precision()
recall = Recall()
accuracy = BinaryAccuracy()

for batch in testing.as_numpy_iterator():
    X, y = batch
    yhat = model.predict(X)
    precision.update_state(y, yhat)
    recall.update_state(y, yhat)
    accuracy.update_state(y, yhat)
print(f'Precision:{precision.result().numpy()}, Recall:{recall.result().numpy()}, Accuracy:{accuracy.result().numpy()}')    

# Saving the Model
model.save(os.path.join('Dog_Vs_Cat\\models','dogcatmodel.keras'))

