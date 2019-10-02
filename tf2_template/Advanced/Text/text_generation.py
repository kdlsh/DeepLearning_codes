# https://www.tensorflow.org/beta/tutorials/text/text_generation

from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
import numpy as np
import os
import time

os.chdir(os.path.dirname(os.path.realpath(__file__)))

## Dataset
shakespeare_url = 'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt'
path_to_file = tf.keras.utils.get_file('shakespeare.txt', shakespeare_url)

text = open(path_to_file, 'rb').read().decode(encoding='utf-8')
# 텍스트의 길이는 그 안에 있는 문자의 수입니다.
print ('텍스트의 길이: {}자'.format(len(text)))
print(text[:250])

# 파일의 고유 문자수를 출력합니다.
vocab = sorted(set(text))
print ('고유 문자수 {}개'.format(len(vocab)))

## Text processing
# 고유 문자에서 인덱스로 매핑 생성
char2idx = {u:i for i, u in enumerate(vocab)}
idx2char = np.array(vocab)

text_as_int = np.array([char2idx[c] for c in text])

# print('{')
# for char,_ in zip(char2idx, range(20)):
#     print('  {:4s}: {:3d},'.format(repr(char), char2idx[char]))
# print('  ...\n}')

# # 텍스트에서 처음 13개의 문자가 숫자로 어떻게 매핑되었는지를 보여줍니다
# print ('{} ---- 문자들이 다음의 정수로 매핑되었습니다 ---- > {}'.format(repr(text[:13]), text_as_int[:13]))

# 단일 입력에 대해 원하는 문장의 최대 길이
seq_length = 100
examples_per_epoch = len(text)//seq_length

# 훈련 샘플/타깃 만들기
char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int)

# for i in char_dataset.take(5):
#     print(idx2char[i.numpy()])

# batch method
sequences = char_dataset.batch(seq_length+1, drop_remainder=True)
for item in sequences.take(5):
    print(repr(''.join(idx2char[item.numpy()])))

# map method
def split_input_target(chunk):
    input_text = chunk[:-1]
    target_text = chunk[1:]
    return input_text, target_text

dataset = sequences.map(split_input_target)

for input_example, target_example in  dataset.take(1):
    print ('입력 데이터: ', repr(''.join(idx2char[input_example.numpy()])))
    print ('타깃 데이터: ', repr(''.join(idx2char[target_example.numpy()])))

for i, (input_idx, target_idx) in enumerate(zip(input_example[:5], target_example[:5])):
    print("{:4d}단계".format(i))
    print("  입력: {} ({:s})".format(input_idx, repr(idx2char[input_idx])))
    print("  예상 출력: {} ({:s})".format(target_idx, repr(idx2char[target_idx])))

## Training batch
# 배치 크기
BATCH_SIZE = 64

# 데이터셋을 섞을 버퍼 크기
# (TF 데이터는 전체 시퀀스를 메모리에 섞지 않는 대신에 버퍼를 유지)
BUFFER_SIZE = 10000

dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)

## Model
# 문자로 된 어휘 사전의 크기
vocab_size = len(vocab)
# 임베딩 차원
embedding_dim = 256
# RNN 유닛(unit) 개수
rnn_units = 1024

def build_model(vocab_size, embedding_dim, rnn_units, batch_size):
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, embedding_dim,
                                batch_input_shape=[batch_size, None]),
        tf.keras.layers.LSTM(rnn_units,
                            return_sequences=True,
                            stateful=True,
                            recurrent_initializer='glorot_uniform'),
        tf.keras.layers.Dense(vocab_size)
    ])
    return model

model = build_model(
    vocab_size = len(vocab),
    embedding_dim=embedding_dim,
    rnn_units=rnn_units,
    batch_size=BATCH_SIZE)

model.summary()

# for input_example_batch, target_example_batch in dataset.take(1):
#     example_batch_predictions = model(input_example_batch)
#     print(example_batch_predictions.shape, "# (배치 크기, 시퀀스 길이, 어휘 사전 크기)")

# sampled_indices = tf.random.categorical(example_batch_predictions[0], num_samples=1)
# sampled_indices = tf.squeeze(sampled_indices,axis=-1).numpy()
# print(sampled_indices)
# print("입력: \n", repr("".join(idx2char[input_example_batch[0]])))
# print()
# print("예측된 다음 문자: \n", repr("".join(idx2char[sampled_indices ])))

## Model training
def loss(labels, logits):
    return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)

# example_batch_loss  = loss(target_example_batch, example_batch_predictions)
# print("예측 배열 크기(shape): ", example_batch_predictions.shape, " # (배치 크기, 시퀀스 길이, 어휘 사전 크기")
# print("스칼라 손실:          ", example_batch_loss.numpy().mean())

model.compile(optimizer='adam', loss=loss)

checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")

checkpoint_callback=tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_prefix,
    save_weights_only=True)

EPOCHS=10

history = model.fit(dataset, epochs=EPOCHS, callbacks=[checkpoint_callback])

## Text generation
model = build_model(vocab_size, embedding_dim, rnn_units, batch_size=1)
model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))
model.build(tf.TensorShape([1, None]))

model.summary()

def generate_text(model, start_string):
    # 평가 단계 (학습된 모델을 사용하여 텍스트 생성)

    # 생성할 문자의 수
    num_generate = 1000

    # 시작 문자열을 숫자로 변환(벡터화)
    input_eval = [char2idx[s] for s in start_string]
    input_eval = tf.expand_dims(input_eval, 0)

    # 결과를 저장할 빈 문자열
    text_generated = []

    # 온도가 낮으면 더 예측 가능한 텍스트가 됩니다.
    # 온도가 높으면 더 의외의 텍스트가 됩니다.
    # 최적의 세팅을 찾기 위한 실험
    temperature = 1.0

    # 여기에서 배치 크기 == 1
    model.reset_states()
    for i in range(num_generate):
        predictions = model(input_eval)
        # 배치 차원 제거
        predictions = tf.squeeze(predictions, 0)

        # 범주형 분포를 사용하여 모델에서 리턴한 단어 예측
        predictions = predictions / temperature
        predicted_id = tf.random.categorical(predictions, num_samples=1)[-1,0].numpy()

        # 예측된 단어를 다음 입력으로 모델에 전달
        # 이전 은닉 상태와 함께
        input_eval = tf.expand_dims([predicted_id], 0)

        text_generated.append(idx2char[predicted_id])

    return (start_string + ''.join(text_generated))

print(generate_text(model, start_string=u"ROMEO: "))

## Advanced (GradientTape)
model = build_model(
    vocab_size = len(vocab),
    embedding_dim=embedding_dim,
    rnn_units=rnn_units,
    batch_size=BATCH_SIZE)

optimizer = tf.keras.optimizers.Adam()

@tf.function
def train_step(inp, target):
    with tf.GradientTape() as tape:
        predictions = model(inp)
        loss = tf.reduce_mean(
            tf.keras.losses.sparse_categorical_crossentropy(
                target, predictions, from_logits=True))
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    return loss

# 훈련 횟수
EPOCHS = 10

for epoch in range(EPOCHS):
    start = time.time()

    # 모든 에포크(Epoch)의 시작에서 은닉 상태를 초기화
    # 초기 은닉 상태는 None
    hidden = model.reset_states()

    for (batch_n, (inp, target)) in enumerate(dataset):
        loss = train_step(inp, target)

        if batch_n % 100 == 0:
            template = '에포크 {} 배치 {} 손실 {}'
            print(template.format(epoch+1, batch_n, loss))

    # 모든 5 에포크(Epoch)마다(체크포인트) 모델 저장
    if (epoch + 1) % 5 == 0:
        model.save_weights(checkpoint_prefix.format(epoch=epoch))

    print ('에포크 {} 손실 {:.4f}'.format(epoch+1, loss))
    print ('1 에포크 당 {}초 소요\n'.format(time.time() - start))

    model.save_weights(checkpoint_prefix.format(epoch=epoch))