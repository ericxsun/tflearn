#!/user/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright @2017 Research, NTC Inc. (ntc.com)
#
# Author: Eric x.sun <eric.x.sun@gmail.com>
#
import random
import struct
import sys
import glob
import codecs

import tensorflow as tf
from tensorflow.core.example import example_pb2
from tensorflow.python.platform import gfile
from utils import LOGGER
from itertools import groupby


FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('command', '', 'build_vocab / binary_to_text / text_to_binary')
tf.app.flags.DEFINE_string('feature_separator', '\t', 'separator between features in source data in format of text')
tf.app.flags.DEFINE_string('in_file', '', 'path to file')
tf.app.flags.DEFINE_string('out_file', '', 'path to file')


# Special tokens
PARAGRAPH_START = '<p>'
PARAGRAPH_END = '</p>'
SENTENCE_START = '<s>'
SENTENCE_END = '</s>'
UNKNOWN_TOKEN = '<UNK>'
PAD_TOKEN = '<PAD>'
DOCUMENT_START = '<d>'
DOCUMENT_END = '</d>'

SPECIAL_TOKENS_FREQ = {
    PARAGRAPH_START: 100000,
    PARAGRAPH_END: 100000,
    SENTENCE_START: 100000,
    SENTENCE_END: 100000,
    DOCUMENT_START: 100000,
    DOCUMENT_END: 100000,
    UNKNOWN_TOKEN: 100000,
    PAD_TOKEN: 100000
}

SPECIAL_TOKENS = set(SPECIAL_TOKENS_FREQ.keys())


class Vocab(object):
    def __init__(self, vocab_file, max_size):
        self._word_to_id = {}
        self._id_to_word = {}
        self._count = 0

        counter = 0
        with open(vocab_file, 'rb') as vocab_f:
            for line in vocab_f:
                counter += 1
                if counter % 1000 == 0:
                    LOGGER.debug("processing line %d", counter)

                pieces = line.split()
                if pieces[0] in self._word_to_id:
                    raise ValueError('duplicated word: %s' % pieces[0])

                if pieces[0] and pieces[0].strip():
                    pieces[0] = pieces[0].strip()

                    self._word_to_id[pieces[0]] = self._count
                    self._id_to_word[self._count] = pieces[0]

                    self._count += 1
                else:
                    sys.stderr.write('bad line: %s\n' % line)

                if self._count > max_size:
                    raise ValueError('too many words: >%d' % max_size)

        assert self.check_vocab(PAD_TOKEN) > 0
        assert self.check_vocab(UNKNOWN_TOKEN) >= 0
        assert self.check_vocab(SENTENCE_START) > 0
        assert self.check_vocab(SENTENCE_END) > 0

    def check_vocab(self, word):
        if word not in self._word_to_id:
            return None

        return self._word_to_id[word]

    def word_to_id(self, word):
        return self._word_to_id.get(word, self._word_to_id[UNKNOWN_TOKEN])

    def id_to_word(self, word_id):
        word = self._id_to_word.get(word_id)
        if word is None:
            raise ValueError('id not found in vocab: %d' % word_id)

        return word

    def num_ids(self):
        return self._count


def gen_example(data_path, num_epochs=None):
    """Generate tf.Examples from path of data files.

    Binary data format: <length><blob>. <length> represents the byte size of <blob>. <blob> is serialized tf.Example
    proto. The tf.Example contains the tokenized article text and summary.

    Args:
        data_path: path pattern of data files, e.g.g, data/data/*.txt.
        num_epochs: number of times to go through the data. None means infinite.

    Yields:
        Deserialized tf.Example.

    If there are multiple files specified, they are accessed randomly.
    """
    epoch = 0

    while True:
        if num_epochs is not None and epoch >= num_epochs:
            break

        file_list = glob.glob(data_path)
        assert file_list, 'empty file_list'

        random.shuffle(file_list)
        for f in file_list:
            reader = open(f, 'rb')
            while True:
                len_bytes = reader.read(8)
                if not len_bytes:
                    break

                str_len = struct.unpack('q', len_bytes)[0]
                example_str = struct.unpack('%ds' % str_len, reader.read(str_len))[0]

                yield example_pb2.Example.FromString(example_str)

        epoch += 1


def pad(ids, pad_id, length):
    """pad or trim list to length

    Args:
        ids: list of ints to be padded.
        pad_id: what to pad with.
        length: length to pad or trim to

    Returns:
        ids trimmed or padded with pad_id
    """

    assert pad_id is not None
    assert length is not None

    if len(ids) < length:
        return ids + [pad_id] * (length - len(ids))
    else:
        return ids[:length]


def convert_words_to_ids(text, vocab, pad_len=None, pad_id=None):
    """convert a sequence of word into a sequence of id.

    Args:
        text: sequence of word.
        vocab: Vocab object.
        pad_len: int, length to pad to.
        pad_id: int, word id for pad symbol.

    Returns:
        a list of ints representing word ids.
    """

    ids = [vocab.word_to_id(w) for w in text.split()]

    if pad_len is not None:
        ids = pad(ids, pad_id, pad_len)

    return ids


def convert_ids_to_words(ids, vocab):
    """Convert a sequence of id into a sequence of word

    Args:
        ids: list of int32.
        vocab: Vocab object.

    Returns:
        list of words corresponding to ids.
    """

    assert isinstance(ids, list), '%s is not a list' % ids

    return [vocab.id_to_word(_id) for _id in ids]


def get_example_feature_words(example, key):
    return example.features[key].bytes_list.value[0]


def convert_paragraph_to_sentences(paragraph, include_token=True):
    def gen_snippet(text, start_tok, end_tok, inclusive=True):
        cur = 0
        while True:
            try:
                start_p = text.index(start_tok, cur)
                end_p = text.index(end_tok, start_p + 1)
                cur = end_p + len(end_tok)

                if inclusive:
                    yield text[start_p:cur]
                else:
                    yield text[start_p + len(start_tok):end_p]
            except ValueError as e:
                raise StopIteration('no more snippets in text: %s' % e)

    s_gen = gen_snippet(paragraph, SENTENCE_START, SENTENCE_END, include_token)
    return [s for s in s_gen]


def convert_text_to_binary():
    """convert text data to binary

    input data format:
    each line looks like:
    article=<d> <p> <s> word1 word2 ... </s> <s> ... </s> </p> ... </d>\tabstract=<d> <p> <s> ... </s> </p> ... </d>
    """
    text_data_path = FLAGS.in_file
    binary_data_path = FLAGS.out_file

    assert text_data_path and binary_data_path, 'filename of text data or binary data should be provided'

    if not gfile.Exists(binary_data_path):
        LOGGER.debug('convert text to binary format: %s => %s', text_data_path, binary_data_path)

        reader = open(text_data_path, mode='rb')
        writer = open(binary_data_path, mode='wb')

        for line in reader:
            tf_example = example_pb2.Example()
            for feature in line.strip().split(FLAGS.feature_separator):
                (k, v) = feature.split('=')
                tf_example.features.feature[k].bytes_list.value.extend([v])

            tf_example_str = tf_example.SerializeToString()
            str_len = len(tf_example_str)
            writer.write(struct.pack('q', str_len))
            writer.write(struct.pack('%ds' % str_len, tf_example_str))

        writer.close()
        reader.close()
    else:
        LOGGER.error('binary data exist: %s', binary_data_path)


def convert_binary_to_text():
    """convert binary data to text

    output data format:
    each line looks like:
    article=<d> <p> <s> word1 word2 ... </s> <s> ... </s> </p> ... </d>\tabstract=<d> <p> <s> ... </s> </p> ... </d>
    """

    binary_data_path = FLAGS.in_file
    text_data_path = FLAGS.out_file

    assert binary_data_path and text_data_path, 'filename of binary data or text data should be provided'

    if not gfile.Exists(text_data_path):
        LOGGER.debug('convert binary to text format: %s => %s', binary_data_path, text_data_path)

        reader = open(binary_data_path, mode='rb')
        writer = codecs.open(text_data_path, mode='wb', encoding='utf-8')

        while True:
            len_bytes = reader.read(8)

            if not len_bytes:
                LOGGER.debug('done reading')
                break

            str_len = struct.unpack('q', len_bytes)[0]
            tf_example_str = struct.unpack('%ds' % str_len, reader.read(str_len))[0]
            tf_example = example_pb2.Example.FromString(tf_example_str)
            examples = []
            for key in tf_example.features.feature:
                value = tf_example.features.feature[key].bytes_list.value[0]
                value = value.decode('utf-8')
                examples.append('%s=%s' % (key, value))

            writer.write('%s\n' % FLAGS.feature_separator.join(examples))

        writer.close()
        reader.close()
    else:
        LOGGER.error('text data exist: %s', text_data_path)


def build_vocab():
    """build vocab from raw data in text format.

    input data format:
    each line looks like:
    article=<d> <p> <s> word1 word2 ... </s> <s> ... </s> </p> ... </d>\tabstract=<d> <p> <s> ... </s> </p> ... </d>
    """

    data_path = FLAGS.in_file
    vocab_path = FLAGS.out_file

    assert data_path and vocab_path, 'filename of data and vocabulary should be provided'

    if not gfile.Exists(vocab_path):
        LOGGER.debug('build vocabulary from %s, storing it into %s', data_path, vocab_path)

        vocab = {}
        counter = 0

        reader = codecs.open(data_path, mode='rb', encoding='utf-8')

        for line in reader:
            counter += 1
            if counter % 1000 == 0:
                LOGGER.debug("processing line %d", counter)

            for feature in line.strip().split(FLAGS.feature_separator):
                (k, v) = feature.split('=')
                word_freq = {k: len(list(g)) for k, g in groupby(sorted(v.split())) if k not in SPECIAL_TOKENS}
                for word, freq in word_freq.items():
                    vocab[word] = vocab.get(word, 0) + freq

        reader.close()

        vocab = sorted(vocab.iteritems(), key=lambda kv: kv[1], reverse=True)
        vocab = [(k, v) for k, v in SPECIAL_TOKENS_FREQ.items()] + vocab

        with gfile.GFile(vocab_path, mode='wb') as vocab_file:
            for word, freq in vocab:
                vocab_file.write(word + b'\t' + str(freq) + b'\n')
    else:
        LOGGER.error('vocabulary file exist: %s', vocab_path)


def main(unused_argv):
    assert FLAGS.command

    if FLAGS.command == 'build_vocab':
        build_vocab()
    elif FLAGS.command == 'build_binary':
        convert_text_to_binary()
    elif FLAGS.command == 'build_text':
        convert_binary_to_text()
    else:
        LOGGER.error('not support command: %s', FLAGS.command)


if __name__ == '__main__':
    tf.app.run(main)
