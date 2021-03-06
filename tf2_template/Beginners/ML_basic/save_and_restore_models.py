# https://www.tensorflow.org/tutorials/keras/save_and_load

# !pip install -q h5py pyyaml

from __future__ import absolute_import, division, print_function, unicode_literals, unicode_literals

import os,sys

import tensorflow as tf
from tensorflow import keras

#tf.__version__

os.chdir(os.path.dirname(os.path.realpath(__file__)))


## Dataset
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

train_labels = train_labels[:1000]
test_labels = test_labels[:1000]

train_images = train_images[:1000].reshape(-1, 28 * 28) / 255.0
test_images = test_images[:1000].reshape(-1, 28 * 28) / 255.0

## Model
# 간단한 Sequential 모델을 반환합니다
def create_model():
  model = tf.keras.models.Sequential([
    keras.layers.Dense(512, activation='relu', input_shape=(784,)),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(10, activation='softmax')
  ])

  model.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])

  return model

checkpoint_path = "training_1/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

# 체크포인트 콜백 만들기
cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)

model = create_model()
# model.summary()

model.fit(train_images, train_labels,  epochs = 10,
          validation_data = (test_images,test_labels),
          callbacks = [cp_callback])

## Untrained model
model = create_model()

loss, acc = model.evaluate(test_images, test_labels)
print("훈련되지 않은 모델의 정확도: {:5.2f}%".format(100*acc))

## Load saved weight
model.load_weights(checkpoint_path)
loss,acc = model.evaluate(test_images, test_labels)
print("복원된 모델의 정확도: {:5.2f}%".format(100*acc))

## Save each 5 epoch weight
# 파일 이름에 에포크 번호를 포함시킵니다(`str.format` 포맷)
checkpoint_path = "training_2/cp-{epoch:04d}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

cp_callback = tf.keras.callbacks.ModelCheckpoint(
    checkpoint_path, verbose=1, save_weights_only=True,
    # 다섯 번째 에포크마다 가중치를 저장합니다
    period=5)

model = create_model()
model.save_weights(checkpoint_path.format(epoch=0))
model.fit(train_images, train_labels,
          epochs = 50, callbacks = [cp_callback],
          validation_data = (test_images,test_labels),
          verbose=0)

## Load saved latest weight
latest = tf.train.latest_checkpoint(checkpoint_dir)
print(latest)

model = create_model()
model.load_weights(latest)
loss, acc = model.evaluate(test_images, test_labels)
print("복원된 모델의 정확도: {:5.2f}%".format(100*acc))

# # 가중치를 저장합니다
# model.save_weights('./checkpoints/my_checkpoint')

# # 가중치를 복원합니다
# model = create_model()
# model.load_weights('./checkpoints/my_checkpoint')

## Save model h5
model = create_model()

model.fit(train_images, train_labels, epochs=5)

# 전체 모델을 HDF5 파일로 저장합니다
model.save('my_model.h5')

## Load model h5
# 가중치와 옵티마이저를 포함하여 정확히 동일한 모델을 다시 생성합니다
new_model = keras.models.load_model('my_model.h5')
new_model.summary()

loss, acc = new_model.evaluate(test_images, test_labels)
print("복원된 모델의 정확도: {:5.2f}%".format(100*acc))