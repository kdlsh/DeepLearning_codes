# https://www.tensorflow.org/guide/data

from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf

import pathlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

np.set_printoptions(precision=4)

## Basic mechanics
dataset = tf.data.Dataset.from_tensor_slices([8, 3, 0, 8, 2, 1])

for elem in dataset: 
    print(elem.numpy())

# iterator (iter, next)
it = iter(dataset)
print(next(it).numpy())

# reduce
print(dataset.reduce(0, lambda state, value: state + value).numpy())

## Dataset structure
dataset1 = tf.data.Dataset.from_tensor_slices(tf.random.uniform([4, 10]))
dataset1.element_spec

dataset2 = tf.data.Dataset.from_tensor_slices(
   (tf.random.uniform([4]),
    tf.random.uniform([4, 100], maxval=100, dtype=tf.int32)))
dataset2.element_spec

dataset3 = tf.data.Dataset.zip((dataset1, dataset2))
dataset3.element_spec

# Dataset containing a sparse tensor.
dataset4 = tf.data.Dataset.from_tensors(tf.SparseTensor(indices=[[0, 0], [1, 2]], values=[1, 2], dense_shape=[3, 4]))
dataset4.element_spec

# Use value_type to see the type of value represented by the element spec
dataset4.element_spec.value_type

dataset1 = tf.data.Dataset.from_tensor_slices(
    tf.random.uniform([4, 10], minval=1, maxval=10, dtype=tf.int32))
dataset1

for z in dataset1: print(z.numpy())

dataset2 = tf.data.Dataset.from_tensor_slices(
   (tf.random.uniform([4]),
    tf.random.uniform([4, 100], maxval=100, dtype=tf.int32)))
dataset2

dataset3 = tf.data.Dataset.zip((dataset1, dataset2))
dataset3

for a, (b,c) in dataset3:
    print('shapes: {a.shape}, {b.shape}, {c.shape}'.format(a=a, b=b, c=c))

## Reading input data
train, test = tf.keras.datasets.fashion_mnist.load_data()

images, labels = train
images = images/255

dataset = tf.data.Dataset.from_tensor_slices((images, labels))
dataset

## Consuming Python generators
def count(stop):
    i = 0
    while i<stop:
        yield i
        i += 1

for n in count(5):
    print(n)

ds_counter = tf.data.Dataset.from_generator(count, args=[25], output_types=tf.int32, output_shapes = (), )

for count_batch in ds_counter.repeat().batch(10).take(10):
    print(count_batch.numpy())

def gen_series():
    i = 0
    while True:
        size = np.random.randint(0,10)
        yield i, np.random.normal(size=(size,))
        i += 1
    
for i,series in gen_series():
    print(i,":",str(series))
    if i>5:
        break

ds_series = tf.data.Dataset.from_generator(
    gen_series, 
    output_types=(tf.int32, tf.float32), 
    output_shapes = ((), (None,))) # () scalar, (,) vector
ds_series

ds_series_batch = ds_series.shuffle(20).padded_batch(10, padded_shapes=([],[None]))

ids, sequence_batch = next(iter(ds_series_batch))
print(ids.numpy())
print()
print(sequence_batch.numpy())

flowers = tf.keras.utils.get_file(
    'flower_photos',
    'https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz',
    untar=True)

# keras ImageDataGenerator
img_gen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255, rotation_range=20)
images, labels = next(img_gen.flow_from_directory(flowers))
print(images.dtype, images.shape)
print(labels.dtype, labels.shape)

# tf.data.Dataset.from_generator
ds = tf.data.Dataset.from_generator(
    img_gen.flow_from_directory, args=[flowers], 
    output_types=(tf.float32, tf.float32), 
    output_shapes = ([32,256,256,3],[32,5])
)
ds

## Consuming TFRecord data
# Creates a dataset that reads all of the examples from two files.
tfrec_url = "https://storage.googleapis.com/download.tensorflow.org/data/fsns-20160927/testdata/fsns-00000-of-00001"
fsns_test_file = tf.keras.utils.get_file("fsns.tfrec", tfrec_url)

dataset = tf.data.TFRecordDataset(filenames = [fsns_test_file ])
dataset

raw_example = next(iter(dataset))
parsed = tf.train.Example.FromString(raw_example.numpy())

parsed.features.feature['image/text']

## Consuming text data
directory_url = 'https://storage.googleapis.com/download.tensorflow.org/data/illiad/'
file_names = ['cowper.txt', 'derby.txt', 'butler.txt']

file_paths = [
    tf.keras.utils.get_file(file_name, directory_url+file_name)
    for file_name in file_names ]

# tf.data.TextLineDataset
dataset = tf.data.TextLineDataset(file_paths)

for line in dataset.take(5):
    print(line.numpy())

# tf.data.Dataset.from_tensor_slices interleave makes it easier to shuffle
files_ds = tf.data.Dataset.from_tensor_slices(file_paths)
lines_ds = files_ds.interleave(tf.data.TextLineDataset, cycle_length=3)

for i, line in enumerate(lines_ds.take(9)):
    if i%3==0:
        print()
    print(line.numpy())


titanic_file = tf.keras.utils.get_file("train.csv", "https://storage.googleapis.com/tf-datasets/titanic/train.csv")
titanic_lines = tf.data.TextLineDataset(titanic_file)

for line in titanic_lines.take(10):
    print(line.numpy())

def survived(line):
    return tf.not_equal(tf.strings.substr(line, 0, 1), "0")

survivors = titanic_lines.skip(1).filter(survived)

for line in survivors.take(10):
    print(line.numpy())

## Consuming CSV data
titanic_file = tf.keras.utils.get_file("train.csv", "https://storage.googleapis.com/tf-datasets/titanic/train.csv")

# pandas
df = pd.DataFrame.from_csv(titanic_file, index_col=None)
df.head()

titanic_slices = tf.data.Dataset.from_tensor_slices(dict(df))

for feature_batch in titanic_slices.take(1):
    for key, value in feature_batch.items():
        print("  {!r:20s}: {}".format(key, value))

# make_csv_dataset; column-type-inference, batching, shuffling etc
titanic_batches = tf.data.experimental.make_csv_dataset(
    titanic_file, batch_size=4,
    label_name="survived")

for feature_batch, label_batch in titanic_batches.take(1):
    print("'survived': {}".format(label_batch))
    print("features:")
    for key, value in feature_batch.items():
        print("  {!r:20s}: {}".format(key, value))

titanic_batches = tf.data.experimental.make_csv_dataset(
    titanic_file, batch_size=4,
    label_name="survived", select_columns=['class','fare','survived'])

for feature_batch, label_batch in titanic_batches.take(1):
    print("'survived': {}".format(label_batch))
    for key, value in feature_batch.items():
        print("  {!r:20s}: {}".format(key, value))

# CsvDataset
titanic_types  = [tf.int32, tf.string, tf.float32, tf.int32, tf.int32, tf.float32, tf.string, tf.string, tf.string, tf.string] 
dataset = tf.data.experimental.CsvDataset(titanic_file, titanic_types , header=True)

for line in dataset.take(10):
    print([item.numpy() for item in line])

# Creates a dataset that reads all of the records from two CSV files, each with
# four float columns which may have missing values.
record_defaults = [999,999,999,999]
dataset = tf.data.experimental.CsvDataset("missing.csv", record_defaults)
dataset = dataset.map(lambda *items: tf.stack(items))
dataset

for line in dataset:
    print(line.numpy())

# Creates a dataset that reads all of the records from two CSV files with
# headers, extracting float data from columns 2 and 4.
record_defaults = [999, 999] # Only provide defaults for the selected columns
dataset = tf.data.experimental.CsvDataset("missing.csv", record_defaults, select_cols=[1,3])
dataset = dataset.map(lambda *items: tf.stack(items))
dataset

## Consuming sets of files
flowers_root = tf.keras.utils.get_file(
    'flower_photos',
    'https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz',
    untar=True)
flowers_root = pathlib.Path(flowers_root)

for item in flowers_root.glob("*"):
    print(item.name)

list_ds = tf.data.Dataset.list_files(str(flowers_root/'*/*'))

for f in list_ds.take(5):
    print(f.numpy())

def process_path(file_path):
    parts = tf.strings.split(file_path, '/')
    return tf.io.read_file(file_path), parts[-2]

labeled_ds = list_ds.map(process_path)

for image_raw, label_text in labeled_ds.take(1):
    print(repr(image_raw.numpy()[:100]))
    print()
    print(label_text.numpy())