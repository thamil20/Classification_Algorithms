import tensorflow as tf
from tensorflow.keras.models import load_model
import os
import numpy as np
import cv2

# Dog Test Image
image = cv2.imread("Dog_Vs_Cat\\out_of_set_data\\dog_test.jpg")
resized_img = tf.image.resize(image, (256,256))

# Cat Test Image
# image = cv2.imread("Dog_Vs_Cat\out_of_set_data\cat_test.jpg")
# resized_img = tf.image.resize(image, (256,256))

# Load the Model
model = load_model(os.path.join('Dog_Vs_Cat\\models','dogcatmodel.keras'))

# Gather Model's prediction
yhat = model.predict(np.expand_dims(resized_img,0))

# Print if the image contains a dog or cat.
if yhat < 0.5:
    print(f'This image is a cat!')
else:
    print(f'This image is a dog!')