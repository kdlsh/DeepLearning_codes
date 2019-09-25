#https://tensorflow.blog/2019/03/06/tensorflow-2-0-keras-api-overview/

from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf 
from tensorflow import keras
from tensorflow.keras import layers

import numpy as np
 
data = np.random.random((1000, 32))
labels = np.random.random((1000, 10))

inputs = tf.keras.Input(shape=(32,))
x = layers.Dense(64, activation='relu')(inputs)
x = layers.Dense(64, activation='relu')(x)
predictions = layers.Dense(10, activation='softmax')(x)

model = tf.keras.Model(inputs=inputs, outputs=predictions)
 
model.compile(optimizer=tf.keras.optimizers.RMSprop(0.001),
loss='categorical_crossentropy',
metrics=['accuracy'])
 
model.fit(data, labels, batch_size=32, epochs=5)