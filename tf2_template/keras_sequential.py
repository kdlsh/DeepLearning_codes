#https://tensorflow.blog/2019/03/06/tensorflow-2-0-keras-api-overview/

from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf 
from tensorflow import keras
from tensorflow.keras import layers

import os

cwd = "D:/workspace/DeepLearning_codes/tf2_template"
log_dir = cwd+"/logs"
weight_dir = cwd+"/weight"
if not os.path.exists(log_dir): os.mkdir(log_dir)
if not os.path.exists(weight_dir): os.mkdir(weight_dir)

## Data
import numpy as np
 
data = np.random.random((1000, 32))
labels = np.random.random((1000, 10))
 
val_data = np.random.random((100, 32))
val_labels = np.random.random((100, 10))

# Dataset object
dataset = tf.data.Dataset.from_tensor_slices((data, labels))
dataset = dataset.batch(32)
 
val_dataset = tf.data.Dataset.from_tensor_slices((val_data, val_labels))
val_dataset = val_dataset.batch(32)

## Model
#model = tf.keras.Sequential()
#model.add(layers.Dense(64, activation='relu'))
#model.add(layers.Dense(64, activation='relu'))
#model.add(layers.Dense(10, activation='softmax'))

model = tf.keras.Sequential([
layers.Dense(64, activation='relu', input_shape=(32,)),
layers.Dense(64, activation='relu'),
layers.Dense(10, activation='softmax')])

#layers.Dense(64, activation='sigmoid')
#layers.Dense(64, activation=tf.keras.activations.sigmoid)
#layers.Dense(64, kernel_regularizer=tf.keras.regularizers.l1(0.01))
#layers.Dense(64, bias_regularizer=tf.keras.regularizers.l2(0.01))
#layers.Dense(64, kernel_initializer='orthogonal')
#layers.Dense(64, bias_initializer=tf.keras.initializers.Constant(2.0))

## Compile
model.compile(optimizer=tf.keras.optimizers.Adam(0.001),
                loss='categorical_crossentropy',
                metrics=['accuracy'])

# regression
#model.compile(optimizer=tf.keras.optimizers.Adam(0.01),
#loss='mse',
#metrics=['mae'])
 
# classification
#model.compile(optimizer=tf.keras.optimizers.RMSprop(0.01),
#loss=tf.keras.losses.CategoricalCrossentropy(),
#metrics=[tf.keras.metrics.CategoricalAccuracy()])

## Fit, Run
#model.fit(data, labels, epochs=10, batch_size=32,
#validation_data=(val_data, val_labels))

callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=2, monitor='val_loss'),
    tf.keras.callbacks.TensorBoard(log_dir=log_dir)
    ]

model.fit(data, labels, batch_size=32, epochs=10, callbacks=callbacks,
validation_data=(val_data, val_labels))

## Save weight and model
#model.save_weights(weight_dir+"/keras_sequential_weight")
model.save_weights(weight_dir+"/keras_sequential_weight.h5", save_format='h5')
#model.load_weights('./weights/my_model')

model.save(cwd+'/keras_sequential_model.h5')
#model = tf.keras.models.load_model('keras_sequential_model.h5')

## Evaluate
model.evaluate(data, labels, batch_size=32)
 
#model.evaluate(dataset, steps=30)

result = model.predict(data, batch_size=32)
print(result.shape)