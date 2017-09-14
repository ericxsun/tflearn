#!/user/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright @2017 Research, NTC Inc. (ntc.com)
#
# Author: Eric x.sun <eric.x.sun@gmail.com>
#
from random import shuffle

import six
import time
import numpy as np
import data
from six.moves import queue as Queue
from six.moves import xrange
from collections import namedtuple
from threading import Thread
import tensorflow as tf
from utils import LOGGER

MODEL_INPUT = namedtuple(
    'model_input',
    'enc_input dec_input target enc_len dec_len origin_article origin_abstract'
)

BUCKET_CACHE_BATCH = 100
QUEUE_NUM_BATCH = 100


class BatchReader(object):
    def __init__(
        self, data_path, vocab, hyper_params, article_key, abstract_key,
        max_article_sentences, max_abstract_sentences, bucketing=True, truncate_input=False
    ):
        """
        Args:
            data_path: tf.Example directory.
            vocab: Vocab object.
            hyper_params: Seq2SeqAttentionModel hyper parameters.
            article_key: article feature key in tf.Example.
            abstract_key: abstract feature key in tf.Example.
            max_article_sentences: maximum number of sentences used from article.
            max_abstract_sentences: maximum number of sentences used from abstract.
            bucketing: whether bucket articles of similar length into the same batch.
            truncate_input: whether to truncate input that is too long.
        """
        self._data_path = data_path
        self._vocab = vocab

        self._batch_size = hyper_params.batch_size
        self._enc_timesteps = hyper_params.enc_timesteps
        self._dec_timesteps = hyper_params.dec_timesteps
        self._min_input_len = hyper_params.min_input_len

        self._article_key = article_key
        self._abstract_key = abstract_key
        self._max_article_sentences = max_article_sentences
        self._max_abstract_sentences = max_abstract_sentences

        self._bucketing = bucketing
        self._truncate_input = truncate_input

        self._input_queue = Queue.Queue(QUEUE_NUM_BATCH * self._batch_size)
        self._bucket_input_queue = Queue.Queue(QUEUE_NUM_BATCH)

        self._input_threads = []
        for _ in xrange(16):
            self._input_threads.append(Thread(target=self._fill_input_queue))
            self._input_threads[-1].daemon = True
            self._input_threads[-1].start()

        self._bucketing_threads = []
        for _ in xrange(4):
            self._bucketing_threads.append(Thread(target=self._fill_bucket_input_queue))
            self._bucketing_threads[-1].daemon = True
            self._bucketing_threads[-1].start()

        self._watch_thread = Thread(target=self._watch_threads)
        self._watch_thread.daemon = True
        self._watch_thread.start()

    def next_batch(self):
        enc_batch = np.zeros((self._batch_size, self._enc_timesteps), dtype=np.int32)
        enc_input_lens = np.zeros(self._batch_size, dtype=np.int32)
        dec_batch = np.zeros((self._batch_size, self._dec_timesteps), dtype=np.int32)
        dec_output_lens = np.zeros(self._batch_size, dtype=np.int32)
        target_batch = np.zeros((self._batch_size, self._dec_timesteps), dtype=np.int32)
        loss_weights = np.zeros((self._batch_size, self._dec_timesteps), dtype=np.float32)
        origin_articles = ['None'] * self._batch_size
        origin_abstracts = ['None'] * self._batch_size

        buckets = self._bucket_input_queue.get()
        for i in xrange(self._batch_size):
            (enc_inputs, dec_inputs, targets, enc_input_len, dec_output_len, article, abstract) = buckets[i]

            origin_articles[i] = article
            origin_abstracts[i] = abstract
            enc_input_lens[i] = enc_input_len
            dec_output_lens[i] = dec_output_len
            enc_batch[i, :] = enc_inputs[:]
            dec_batch[i, :] = dec_inputs[:]
            target_batch[i, :] = targets[:]

            for j in xrange(dec_output_len):
                loss_weights[i][j] = 1

        return (
            enc_batch, dec_batch, target_batch, enc_input_lens, dec_output_lens, loss_weights,
            origin_articles, origin_abstracts
        )

    def _fill_input_queue(self):
        sentence_start_id = self._vocab.word_to_id(data.SENTENCE_START)
        sentence_end_id = self._vocab.word_to_id(data.SENTENCE_END)
        pad_id = self._vocab.word_to_id(data.PAD_TOKEN)

        input_gen = self._gen_text(data.gen_example(self._data_path))

        while True:
            (article, abstract) = six.next(input_gen)
            article_sentences = [sent.strip() for sent in data.convert_paragraph_to_sentences(article, False)]
            abstract_sentences = [sent.strip() for sent in data.convert_paragraph_to_sentences(abstract, False)]

            enc_inputs = []
            dec_inputs = [sentence_start_id]  # use the <s> as the <GO> symbol for decoder inputs

            # convert first N sentences to word ids, stripping existing <s> and </s>
            for i in xrange(min(self._max_article_sentences, len(article_sentences))):
                enc_inputs += data.convert_words_to_ids(article_sentences[i], self._vocab)

            for i in xrange(min(self._max_abstract_sentences, len(abstract_sentences))):
                dec_inputs += data.convert_words_to_ids(abstract_sentences[i], self._vocab)

            # filter out too-short input
            if len(enc_inputs) < self._min_input_len or len(dec_inputs) < self._min_input_len:
                LOGGER.warn('drop an example - too short. enc: %d, dec: %d', len(enc_inputs), len(dec_inputs))
                continue

            if not self._truncate_input:
                if len(enc_inputs) > self._enc_timesteps or len(dec_inputs) > self._dec_timesteps:
                    LOGGER.warn('drop an example - too long. enc: %d, dec: %d', len(enc_inputs), len(dec_inputs))
                    continue
            else:
                if len(enc_inputs) > self._enc_timesteps:
                    enc_inputs = enc_inputs[:self._enc_timesteps]

                if len(dec_inputs) > self._dec_timesteps:
                    dec_inputs = dec_inputs[:self._dec_timesteps]

            # targets is dec_inputs without <s> at beginning, plus </s> at end
            targets = dec_inputs[1:]
            targets.append(sentence_end_id)

            enc_input_len = len(enc_inputs)
            dec_output_len = len(targets)

            # pad if necessary
            enc_inputs += [pad_id] * (self._enc_timesteps - len(enc_inputs))
            dec_inputs += [sentence_end_id] * (self._dec_timesteps - len(dec_inputs))
            targets += [sentence_end_id] * (self._dec_timesteps - len(targets))

            # 'enc_input dec_input target enc_len dec_len origin_article origin_abstract'
            element = MODEL_INPUT(
                enc_input=enc_inputs,
                dec_input=dec_inputs,
                target=targets,
                enc_len=enc_input_len,
                dec_len=dec_output_len,
                origin_article=' '.join(article_sentences),
                origin_abstract=' '.join(abstract_sentences)
            )

            self._input_queue.put(element)

    def _fill_bucket_input_queue(self):
        while True:
            inputs = []

            for _ in xrange(self._batch_size * BUCKET_CACHE_BATCH):
                inputs.append(self._input_queue.get())

            if self._bucketing:
                inputs = sorted(inputs, key=lambda inp: inp.enc_len)

            batches = []
            for i in xrange(0, len(inputs), self._batch_size):
                batches.append(inputs[i:i + self._batch_size])

            shuffle(batches)

            for b in batches:
                self._bucket_input_queue.put(b)

    def _watch_threads(self):
        while True:
            time.sleep(60)

            input_threads = []
            for t in self._input_threads:
                if t.is_alive():
                    input_threads.append(t)
                else:
                    LOGGER.error('found input thread dead')
                    new_t = Thread(target=self._fill_input_queue)
                    input_threads.append(new_t)
                    input_threads[-1].daemon = True
                    input_threads[-1].start()

            self._input_threads = input_threads

            bucketing_threads = []
            for t in self._bucketing_threads:
                if t.is_alive():
                    bucketing_threads.append(t)
                else:
                    LOGGER.error('found bucketing thread dead')
                    new_t = Thread(target=self._fill_bucket_input_queue)
                    bucketing_threads.append(new_t)
                    bucketing_threads[-1].daemon = True
                    bucketing_threads[-1].start()

            self._bucketing_threads = bucketing_threads

    def _gen_text(self, example):
        while True:
            ex = six.next(example)
            try:
                article_text = self._get_example_feature_text(ex, self._article_key)
                abstract_text = self._get_example_feature_text(ex, self._abstract_key)
            except ValueError:
                LOGGER.error('failed to get article or abstract from example')
                continue

            yield (article_text, abstract_text)

    @staticmethod
    def _get_example_feature_text(example, key):
        return example.features.feature[key].bytes_list.value[0]
