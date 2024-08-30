# This program builds and saves an image classification model.
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"]="2"
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

data_dir = "Wild_Animal_Multi-Class\\dataset"
class_names = ["cheetah", "fox", "hyena", "lion", "tiger", "wolf"]

dataset = tf.keras.utils.image_dataset_from_directory(
    data_dir, 
    batch_size=32,
    labels='inferred',
    label_mode='categorical',
    class_names=class_names,
    color_mode='rgba',
    image_size=(256,256))

data_iterator = dataset.as_numpy_iterator()
batch = data_iterator.next()

print(batch[0].shape)
print(batch[1].shape)
print(batch[1])

scaled = batch[0]/255

dataset = dataset.map(lambda x,y: (x/255,y))
def plotting(batch):
    fig, ax, = plt.subplots(ncols=4, figsize=(20,20))
    for idx, img in enumerate(batch[0][:4]):
        ax[idx].imshow(img.astype(int))
        ax[idx].title.set_text(batch[1][idx])
    plt.show()
    
plotting(batch)