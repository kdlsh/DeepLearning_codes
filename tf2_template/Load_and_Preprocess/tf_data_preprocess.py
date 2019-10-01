# https://www.tensorflow.org/guide/data

from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf

import pathlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

np.set_printoptions(precision=4)


## Batching dataset elements
inc_dataset = tf.data.Dataset.range(100)
dec_dataset = tf.data.Dataset.range(0, -100, -1)
dataset = tf.data.Dataset.zip((inc_dataset, dec_dataset))
batched_dataset = dataset.batch(4)

it = iter(batched_dataset)
for batch in batched_dataset.take(4):
    print([arr.numpy() for arr in batch])

batched_dataset = dataset.batch(7, drop_remainder=True)
batched_dataset

dataset = tf.data.Dataset.range(100)
dataset = dataset.map(lambda x: tf.fill([tf.cast(x, tf.int32)], x))
dataset = dataset.padded_batch(4, padded_shapes=(None,))

for batch in dataset.take(2):
    print(batch.numpy())
    print()

## Training workflows
titanic_url = "https://storage.googleapis.com/tf-datasets/titanic/train.csv"
titanic_file = tf.keras.utils.get_file("train.csv", titanic_url)
titanic_lines = tf.data.TextLineDataset(titanic_file)

def plot_batch_sizes(ds):
    batch_sizes = [batch.shape[0] for batch in ds]
    plt.bar(range(len(batch_sizes)), batch_sizes)
    plt.xlabel('Batch number')
    plt.ylabel('Batch size')
    plt.show()

# repeat(epoch), batch(batch_size)
titanic_batches = titanic_lines.repeat(3).batch(128)
# plot_batch_sizes(titanic_batches)

# clear epoch separation
titanic_batches = titanic_lines.batch(128).repeat(3)
# plot_batch_sizes(titanic_batches)

# epoch iteration (custom computation)
epochs = 3
dataset = titanic_lines.batch(128)

for epoch in range(epochs):
    for batch in dataset:
        print(batch.shape)
    print("End of epoch: ", epoch)

# Randomly shuffling input data
lines = tf.data.TextLineDataset(titanic_file)
counter = tf.data.experimental.Counter()

dataset = tf.data.Dataset.zip((counter, lines))
dataset = dataset.shuffle(buffer_size=100)
dataset = dataset.batch(20)
dataset

n,line_batch = next(iter(dataset))
print(n.numpy())

dataset = tf.data.Dataset.zip((counter, lines))
shuffled = dataset.shuffle(buffer_size=100).batch(10).repeat(2)

print("Here are the item ID's near the epoch boundary:\n")
for n, line_batch in shuffled.skip(60).take(5):
    print(n.numpy())

shuffle_repeat = [n.numpy().mean() for n, line_batch in shuffled]
# plt.plot(shuffle_repeat, label="shuffle().repeat()")
# plt.ylabel("Mean item ID")
# plt.legend()
# plt.show()

dataset = tf.data.Dataset.zip((counter, lines))
shuffled = dataset.repeat(2).shuffle(buffer_size=100).batch(10)

print("Here are the item ID's near the epoch boundary:\n")
for n, line_batch in shuffled.skip(55).take(15):
    print(n.numpy())

repeat_shuffle = [n.numpy().mean() for n, line_batch in shuffled]
# plt.plot(shuffle_repeat, label="shuffle().repeat()")
# plt.plot(repeat_shuffle, label="repeat().shuffle()")
# plt.ylabel("Mean item ID")
# plt.legend()
# plt.show()

## Preprocessing data
flowers_root = tf.keras.utils.get_file(
    'flower_photos',
    'https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz',
    untar=True)
flowers_root = pathlib.Path(flowers_root)
list_ds = tf.data.Dataset.list_files(str(flowers_root/'*/*'))

# Reads an image from a file, decodes it into a dense tensor, and resizes it
# to a fixed shape.
def parse_image(file_path):
    parts = tf.strings.split(file_path, '\\')
    label = parts[-2]

    image = tf.io.read_file(file_path)
    image = tf.image.decode_jpeg(image)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, [128, 128])
    return image, label

file_path = next(iter(list_ds))
image, label = parse_image(file_path)

def show(image, label):
    plt.figure()
    plt.imshow(image)
    plt.title(label.numpy().decode('utf-8'))
    plt.axis('off')
    plt.show()

images_ds = list_ds.map(parse_image)
# for image, label in images_ds.take(2):
#     show(image, label)

## Applying arbitrary Python logic (tf.py_function)
import scipy.ndimage as ndimage

def random_rotate_image(image):
    image =  ndimage.rotate(image, np.random.uniform(-30,30), reshape=False)
    return image

image, label = next(iter(images_ds))
image = random_rotate_image(image)
# show(image, label)

def tf_random_rotate_image(image, label):
    im_shape = image.shape
    [image,]= tf.py_function(random_rotate_image, [image], [tf.float32])
    image.set_shape(im_shape)
    return image, label

rot_ds = images_ds.map(tf_random_rotate_image)

# for image, label in rot_ds.take(2):
#     show(image, label)

## Parsing tf.Example protocol buffer messages
fsns_url = "https://storage.googleapis.com/download.tensorflow.org/data/fsns-20160927/testdata/fsns-00000-of-00001"
fsns_test_file = tf.keras.utils.get_file("fsns.tfrec", fsns_url)
dataset = tf.data.TFRecordDataset(filenames = [fsns_test_file ])

raw_example = next(iter(dataset))
parsed = tf.train.Example.FromString(raw_example.numpy())

feature = parsed.features.feature
raw_img = feature['image/encoded'].bytes_list.value[0]
img = tf.image.decode_png(raw_img)
# plt.imshow(img)
# plt.axis('off')
# _ = plt.title(feature["image/text"].bytes_list.value[0])
# plt.show()

raw_example = next(iter(dataset))

def tf_parse(raw_examples):
    example = tf.io.parse_example(
        raw_example[tf.newaxis], 
        {'image/encoded':tf.io.FixedLenFeature(shape=(),dtype=tf.string),
        'image/text':tf.io.FixedLenFeature(shape=(), dtype=tf.string)})
    return example['image/encoded'][0], example['image/text'][0]

img, txt = tf_parse(raw_example)
# print(txt.numpy())
# print(repr(img.numpy()[:20]), "...")

decoded = dataset.map(tf_parse)
image_batch, text_batch = next(iter(decoded.batch(10)))
image_batch.shape

## Time series windowing
range_ds = tf.data.Dataset.range(100000)

batches = range_ds.batch(10, drop_remainder=True)
for batch in batches.take(5):
    print(batch.numpy())

# dense 1 step
def dense_1_step(batch):
    # Shift features and labels one step relative to each other.
    return batch[:-1], batch[1:]

predict_dense_1_step = batches.map(dense_1_step)

for features, label in predict_dense_1_step.take(3):
    print(features.numpy(), " => ", label.numpy())

# next 5 steps
batches = range_ds.batch(15, drop_remainder=True)

def label_next_5_steps(batch):
  return (batch[:-5],   # Take the first 5 steps
          batch[-5:])   # take the remainder

predict_5_steps = batches.map(label_next_5_steps)

for features, label in predict_5_steps.take(3):
    print(features.numpy(), " => ", label.numpy())

# allow some overlab
feature_length = 10
label_length = 5

features = range_ds.batch(feature_length, drop_remainder=True)
labels = range_ds.batch(feature_length).skip(1).map(lambda labels: labels[:-5])

predict_5_steps = tf.data.Dataset.zip((features, labels))

for features, label in predict_5_steps.take(3):
    print(features.numpy(), " => ", label.numpy())

