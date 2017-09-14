#!/user/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright @2017 Research, NTC Inc. (ntc.com)
#
# Author: Eric x.sun <eric.x.sun@gmail.com>
#

import os
import tensorflow as tf
import time
import data

import beam_search


FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer('max_decoder_steps', 100000, 'maximum number of decoding steps')
tf.app.flags.DEFINE_integer('decode_batches_per_ckpt', 8000, 'number of batches to decode before restoring next checkpoint')

DECODE_LOOP_DELAY_SECS = 60
DECODE_IO_FLUSH_INTERVAL = 100


class DecodeIO(object):
    """Writes the decoded and references to RKV files for Rouge score."""

    def __init__(self, out_dir):
        self._cnt = 0
        self._out_dir = out_dir

        if not os.path.exists(self._out_dir):
            os.mkdir(self._out_dir)

        self._ref_file = None
        self._decode_file = None

    def write(self, reference, decode):
        """Writes the reference and decoded outputs to RKV files.

        Args:
          reference: The human (correct) result.
          decode: The machine-generated result
        """
        self._ref_file.write('output=%s\n' % reference)
        self._decode_file.write('output=%s\n' % decode)
        self._cnt += 1
        if self._cnt % DECODE_IO_FLUSH_INTERVAL == 0:
            self._ref_file.flush()
            self._decode_file.flush()

    def reset_files(self):
        """Resets the output files. Must be called once before Write()."""

        if self._ref_file: self._ref_file.close()
        if self._decode_file: self._decode_file.close()

        timestamp = int(time.time())
        self._ref_file = open(os.path.join(self._out_dir, 'ref%d' % timestamp), 'w')
        self._decode_file = open(os.path.join(self._out_dir, 'decode%d' % timestamp), 'w')


class BeamSearchDecoder(object):
    def __init__(self, model, batch_reader, hyper_params, vocab):
        self._model = model
        self._model.build_graph()

        self._batch_reader = batch_reader
        self._hyper_params = hyper_params
        self._vocab = vocab

        self._saver = tf.train.Saver()
        self._decode_io = DecodeIO(FLAGS.decode_dir)

    def decode_loop(self):
        """Decoding loop for long running process."""
        sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
        step = 0
        while step < FLAGS.max_decode_steps:
            time.sleep(DECODE_LOOP_DELAY_SECS)
            if not self._decode(self._saver, sess):
                continue
            step += 1

    def _decode(self, saver, sess):
        """Restore a checkpoint and decode it.

        Args:
          saver: Tensorflow checkpoint saver.
          sess: Tensorflow session.
        Returns:
          if success, returns true, otherwise, false.
        """
        ckpt_state = tf.train.get_checkpoint_state(FLAGS.log_root)
        if not (ckpt_state and ckpt_state.model_checkpoint_path):
            tf.logging.info('No model to decode yet at %s', FLAGS.log_root)
            return False

        tf.logging.info('checkpoint path %s', ckpt_state.model_checkpoint_path)
        ckpt_path = os.path.join(FLAGS.log_root, os.path.basename(ckpt_state.model_checkpoint_path))
        tf.logging.info('renamed checkpoint path %s', ckpt_path)
        saver.restore(sess, ckpt_path)

        self._decode_io.reset_files()
        for _ in xrange(FLAGS.decode_batches_per_ckpt):
            (
                article_batch, _, _, article_lens, _, _, origin_articles, origin_abstracts
            ) = self._batch_reader.next_batch()

            for i in xrange(self._hyper_params.batch_size):
                bs = beam_search.BeamSearch(
                    self._model, self._hyper_params.batch_size,
                    self._vocab.WordToId(data.SENTENCE_START),
                    self._vocab.WordToId(data.SENTENCE_END),
                    self._hyper_params.dec_timesteps
                )

                article_batch_cp = article_batch.copy()
                article_batch_cp[:] = article_batch[i:i + 1]
                article_lens_cp = article_lens.copy()
                article_lens_cp[:] = article_lens[i:i + 1]
                best_beam = bs.beam_search(sess, article_batch_cp, article_lens_cp)[0]
                decode_output = [int(t) for t in best_beam.tokens[1:]]
                self._decode_batch(origin_articles[i], origin_abstracts[i], decode_output)
        return True

    def _decode_batch(self, article, abstract, output_ids):
        """Convert id to words and writing results.

        Args:
          article: The original article string.
          abstract: The human (correct) abstract string.
          output_ids: The abstract word ids output by machine.
        """
        decoded_output = ' '.join(data.convert_ids_to_words(output_ids, self._vocab))
        end_p = decoded_output.find(data.SENTENCE_END, 0)
        if end_p != -1:
            decoded_output = decoded_output[:end_p]
        tf.logging.info('article:  %s', article)
        tf.logging.info('abstract: %s', abstract)
        tf.logging.info('decoded:  %s', decoded_output)
        self._decode_io.write(abstract, decoded_output.strip())
