# https://www.tensorflow.org/tutorials/images/transfer_learning_with_hub

from __future__ import absolute_import, division, print_function, unicode_literals

import os
import matplotlib.pylab as plt
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras import layers

os.chdir(os.path.dirname(os.path.realpath(__file__)))

## hub model
classifier_url ="https://tfhub.dev/google/tf2-preview/mobilenet_v2/classification/2" #@param {type:"string"}

IMAGE_SHAPE = (224, 224)

classifier = tf.keras.Sequential([
    hub.KerasLayer(classifier_url, input_shape=IMAGE_SHAPE+(3,))
])

## Dataset (single image)
import numpy as np
import PIL.Image as Image

grace_hopper_jpg = 'https://storage.googleapis.com/download.tensorflow.org/example_images/grace_hopper.jpg'
grace_hopper = tf.keras.utils.get_file('image.jpg', grace_hopper_jpg)
grace_hopper = Image.open(grace_hopper).resize(IMAGE_SHAPE)
# grace_hopper.show()
# grace_hopper.save("grace_hopper.jpg")

grace_hopper = np.array(grace_hopper)/255.0
print(grace_hopper.shape)

## Predict (sigle image)
result = classifier.predict(grace_hopper[np.newaxis, ...])
print(result.shape)

predicted_class = np.argmax(result[0], axis=-1)
print(predicted_class)

## Decode the predictions
labels_txt = 'https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt'
labels_path = tf.keras.utils.get_file('ImageNetLabels.txt', labels_txt)
imagenet_labels = np.array(open(labels_path).read().splitlines())

plt.figure(figsize=(10,10))
plt.imshow(grace_hopper)
plt.axis('off')
predicted_class_name = imagenet_labels[predicted_class]
_ = plt.title("Prediction: " + predicted_class_name.title())
plt.show()