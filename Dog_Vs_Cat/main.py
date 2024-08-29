import tensorflow as tf
import os
import numpy as np
from matplotlib import pyplot as plt

# Limit GPUs memory growth, avoiding OOM errors.
#gpus = tf.config.experimental.list_physical_devices('GPU')
#for gpu in gpus:
#    tf.config.experimental.set_memory_growth(gpu, True)

# Remove unnecessary images.
data_dir = "Dog_Vs_Cat\\initial_dataset"
image_exts = ["png"]

for image_class in os.listdir(data_dir):
    for image in os.listdir(os.path.join(data_dir, image_class)):
        image_path = os.path.join(data_dir, image_class, image)
        pass
    

data = tf.keras.utils.image_dataset_from_directory(data_dir)
data_iterator = data.as_numpy_iterator()
batch = data_iterator.next()
