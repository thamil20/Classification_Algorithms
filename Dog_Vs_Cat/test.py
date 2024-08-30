# WORK IN PROGRESS

import tensorflow as tf
from tensorflow.keras.models import load_model
import os
import numpy as np
import cv2
from random import shuffle

""""
dog_imgs = os.listdir("Dog_Vs_Cat\\test_script_data\\clean_dog_data")
print(dog_imgs)
resized_dog_imgs = []
for dog_img in dog_imgs:
    # Dog Test Image
    resized_img = tf.image.resize(dog_img, (256,256))
    resized_dog_imgs.append(resized_img)

cat_imgs = os.listdir("Dog_Vs_Cat\\test_script_data\\clean_dog_data")
resized_cat_imgs = []
for cat_img in cat_imgs:
    # Cat Test Image
    resized_img = tf.image.resize(cat_img, (256,256))
    resized_cat_imgs.append(resized_img)


#resized_imgs = resized_dog_imgs + resized_cat_imgs
#resized_imgs = resized_imgs.shuffle()
#for resized_img in resized_imgs:
"""
# Dog Test Image
#image = cv2.imread("Dog_Vs_Cat\\out_of_set_data\\dog_test.jpg")
#resized_img = tf.image.resize(image, (256,256))

# Cat Test Image
image = cv2.imread("Dog_Vs_Cat\out_of_set_data\cat_test.jpg")
resized_img = tf.image.resize(image, (256,256))



# Load the Model
model = load_model("Dog_Vs_Cat\\models\\dogcatmodel.keras")


# Gather Model's prediction
yhat = model.predict(np.expand_dims(resized_img,0))

    #Print if the image contains a dog or cat.
if yhat < 0.5:
    print(f'This image is a cat!')
else:
    print(f'This image is a dog!')