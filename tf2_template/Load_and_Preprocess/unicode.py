# https://www.tensorflow.org/beta/tutorials/text/unicode

from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf

## tf.string datatype
print(tf.constant(u"Thanks 😊"))
print(tf.constant([u"You're", u"welcome!"]).shape)

## Unicode expression
# UTF-8로 인코딩된 string 스칼라로 표현한 유니코드 문자열입니다.
text_utf8 = tf.constant(u"语言处理")
print(text_utf8)

# UTF-16-BE로 인코딩된 string 스칼라로 표현한 유니코드 문자열입니다.
text_utf16be = tf.constant(u"语言处理".encode("UTF-16-BE"))
print(text_utf16be)

# 유니코드 코드 포인트의 벡터로 표현한 유니코드 문자열입니다.
text_chars = tf.constant([ord(char) for char in u"语言处理"])
print(text_chars)

## 변환
tf.strings.unicode_decode(text_utf8, input_encoding='UTF-8')
tf.strings.unicode_encode(text_chars, output_encoding='UTF-8')
tf.strings.unicode_transcode(text_utf8,
                            input_encoding='UTF8',
                            output_encoding='UTF-16-BE')

## batch dimension
# UTF-8 인코딩된 문자열로 표현한 유니코드 문자열의 배치입니다. 
batch_utf8 = [s.encode('UTF-8') for s in
                [u'hÃllo',  u'What is the weather tomorrow',  u'Göödnight', u'😊']]
batch_chars_ragged = tf.strings.unicode_decode(batch_utf8, input_encoding='UTF-8')
for sentence_chars in batch_chars_ragged.to_list():
    print(sentence_chars)

# tf.RaggedTensor를 바로 사용하거나, 
# 패딩(padding)을 사용해 tf.Tensor로 변환하거나, 
# tf.RaggedTensor.to_tensor 와 tf.RaggedTensor.to_sparse 사용해 tf.SparseTensor로 변환가능
batch_chars_padded = batch_chars_ragged.to_tensor(default_value=-1)
print(batch_chars_padded.numpy())

batch_chars_sparse = batch_chars_ragged.to_sparse()

# 길이가 같은 문자열 인코딩 tf.Tensor
tf.strings.unicode_encode([[99, 97, 116], [100, 111, 103], [ 99, 111, 119]],
                          output_encoding='UTF-8')

# 길이가 다른 문자열 인코딩 tf.RaggedTensor
tf.strings.unicode_encode(batch_chars_ragged, output_encoding='UTF-8')

# padded or sparse to tf.RaggedTensor for unicode_encode
tf.strings.unicode_encode(
    tf.RaggedTensor.from_sparse(batch_chars_sparse),
    output_encoding='UTF-8')
tf.strings.unicode_encode(
    tf.RaggedTensor.from_tensor(batch_chars_padded, padding=-1),
    output_encoding='UTF-8')

## Unicode methods
# tf.strings.length
# UTF8에서 마지막 문자는 4바이트를 차지합니다.
thanks = u'Thanks 😊'.encode('UTF-8')
num_bytes = tf.strings.length(thanks).numpy()
num_chars = tf.strings.length(thanks, unit='UTF8_CHAR').numpy()
print('{} 바이트; {}개의 UTF-8 문자'.format(num_bytes, num_chars))

# tf.strings.substr
# 기본: unit='BYTE'. len=1이면 바이트 하나를 반환합니다.
tf.strings.substr(thanks, pos=7, len=1).numpy()

# unit='UTF8_CHAR'로 지정하면 4 바이트인 문자 하나를 반환합니다.
print(tf.strings.substr(thanks, pos=7, len=1, unit='UTF8_CHAR').numpy())

# tf.strings.unicode_split
tf.strings.unicode_split(thanks, 'UTF-8').numpy()

# tf.strings.unicode_decode_with_offsets
codepoints, offsets = tf.strings.unicode_decode_with_offsets(u"🎈🎉🎊", 'UTF-8')
for (codepoint, offset) in zip(codepoints.numpy(), offsets.numpy()):
    print("바이트 오프셋 {}: 코드 포인트 {}".format(offset, codepoint))

## Unicode script
uscript = tf.strings.unicode_script([33464, 1041])  # ['芸', 'Б']
print(uscript.numpy())  # [17, 8] == [USCRIPT_HAN, USCRIPT_CYRILLIC]
print(tf.strings.unicode_script(batch_chars_ragged))

## Example (simple segmentation)

# dtype: string; shape: [num_sentences]
# 처리할 문장들 입니다. 이 라인을 수정해서 다른 입력값을 시도해 보세요!
sentence_texts = [u'Hello, world.', u'世界こんにちは']

# sentence_char_codepoint[i, j]는
# i번째 문장 안에 있는 j번째 문자에 대한 코드 포인트 입니다.
sentence_char_codepoint = tf.strings.unicode_decode(sentence_texts, 'UTF-8')
print(sentence_char_codepoint)

# sentence_char_codepoint[i, j]는 
# i번째 문장 안에 있는 j번째 문자의 유니코드 스크립트 입니다.
sentence_char_script = tf.strings.unicode_script(sentence_char_codepoint)
print(sentence_char_script)

# sentence_char_starts_word[i, j]는 
# i번째 문장 안에 있는 j번째 문자가 단어의 시작이면 True 입니다.
sentence_char_starts_word = tf.concat(
    [tf.fill([sentence_char_script.nrows(), 1], True),
     tf.not_equal(sentence_char_script[:, 1:], sentence_char_script[:, :-1])],
    axis=1)

# word_starts[i]은 (모든 문장의 문자를 일렬로 펼친 리스트에서)
# i번째 단어가 시작되는 문자의 인덱스 입니다.
word_starts = tf.squeeze(tf.where(sentence_char_starts_word.values), axis=1)
print(word_starts)

# word_char_codepoint[i, j]은 
# i번째 단어 안에 있는 j번째 문자에 대한 코드 포인트 입니다.
word_char_codepoint = tf.RaggedTensor.from_row_starts(
    values=sentence_char_codepoint.values,
    row_starts=word_starts)
print(word_char_codepoint)

# sentence_num_words[i]는 i번째 문장 안에 있는 단어의 수입니다.
sentence_num_words = tf.reduce_sum(
    tf.cast(sentence_char_starts_word, tf.int64),
    axis=1)

# sentence_word_char_codepoint[i, j, k]는 i번째 문장 안에 있는
# j번째 단어 안의 k번째 문자에 대한 코드 포인트입니다.
sentence_word_char_codepoint = tf.RaggedTensor.from_row_lengths(
    values=word_char_codepoint,
    row_lengths=sentence_num_words)
print(sentence_word_char_codepoint)

print(tf.strings.unicode_encode(sentence_word_char_codepoint, 'UTF-8').to_list())