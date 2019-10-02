# https://www.tensorflow.org/tutorials/structured_data/feature_columns

#!pip install -q sklearn

from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import feature_column
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split

URL = 'https://storage.googleapis.com/applied-dl/heart.csv'
dataframe = pd.read_csv(URL)
#print(dataframe.head())

train, test = train_test_split(dataframe, test_size=0.2)
train, val = train_test_split(train, test_size=0.2)
print(len(train), '훈련 샘플')
print(len(val), '검증 샘플')
print(len(test), '테스트 샘플')

# 판다스 데이터프레임으로부터 tf.data 데이터셋을 만들기 위한 함수
def df_to_dataset(dataframe, shuffle=True, batch_size=32):
    dataframe = dataframe.copy()
    labels = dataframe.pop('target')
    ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(dataframe))
    ds = ds.batch(batch_size)
    return ds

batch_size = 32 # 5 예제를 위해 작은 배치 크기를 사용합니다.
train_ds = df_to_dataset(train, batch_size=batch_size)
val_ds = df_to_dataset(val, shuffle=False, batch_size=batch_size)
test_ds = df_to_dataset(test, shuffle=False, batch_size=batch_size)

for feature_batch, label_batch in train_ds.take(1):
    print('전체 특성:', list(feature_batch.keys()))
    print('나이 특성의 배치:', feature_batch['age'])
    print('타깃의 배치:', label_batch )

# 특성 열을 시험해 보기 위해 샘플 배치를 만듭니다.
example_batch = next(iter(train_ds))[0]

# 특성 열을 만들고 배치 데이터를 변환하는 함수
def demo(feature_column):
    feature_layer = layers.DenseFeatures(feature_column)
    print(feature_layer(example_batch).numpy())

## build feature layer
feature_columns = []

# 수치형 열
for header in ['age', 'trestbps', 'chol', 'thalach', 'oldpeak', 'slope', 'ca']:
    feature_columns.append(feature_column.numeric_column(header))

# 버킷형 열 ordinal
age = feature_column.numeric_column("age")
# demo(age)
age_buckets = feature_column.bucketized_column(age, boundaries=[18, 25, 30, 35, 40, 45, 50, 55, 60, 65])
# demo(age_buckets)
feature_columns.append(age_buckets)

# 범주형 열 sparse categorical
thal = feature_column.categorical_column_with_vocabulary_list(
        'thal', ['fixed', 'normal', 'reversible'])
thal_one_hot = feature_column.indicator_column(thal)
#demo(thal_one_hot)
feature_columns.append(thal_one_hot)

# 임베딩 열 dense categorical
thal_embedding = feature_column.embedding_column(thal, dimension=8)
#demo(thal_embedding)
feature_columns.append(thal_embedding)

# # categorical - hash
# thal_hashed = feature_column.categorical_column_with_hash_bucket(
#       'thal', hash_bucket_size=1000)
# demo(feature_column.indicator_column(thal_hashed))

# 교차 특성 열
crossed_feature = feature_column.crossed_column([age_buckets, thal], hash_bucket_size=1000)
crossed_feature = feature_column.indicator_column(crossed_feature)
#demo(crossed_feature)
feature_columns.append(crossed_feature)

feature_layer = tf.keras.layers.DenseFeatures(feature_columns)


model = tf.keras.Sequential([
    feature_layer,
    layers.Dense(128, activation='relu'),
    layers.Dense(128, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam',
                loss='binary_crossentropy',
                metrics=['accuracy'])

model.fit(train_ds,
            validation_data=val_ds,
            epochs=5)

loss, accuracy = model.evaluate(test_ds)
print("정확도", accuracy)