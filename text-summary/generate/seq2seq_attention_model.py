#!/user/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright @2017 Research, NTC Inc. (ntc.com)
#
# Author: Eric x.sun <eric.x.sun@gmail.com>
#

import tensorflow as tf
import numpy as np
import seq2seq_lib
from collections import namedtuple


HYPER_PARAMS = namedtuple(
    'hyper_params',
    'mode, batch_size, num_hidden, emb_dim, enc_layers, enc_timesteps, dec_timesteps, '
    'max_grad_norm, num_softmax_samples, min_input_len, min_lr, lr'
)


class Seq2SeqAttentionModel(object):
    def __init__(self, hyper_params, vocab, num_gpus=0):
        self._hyper_params = hyper_params
        self._vocab = vocab
        self._num_gpus = num_gpus
        self._cur_gpu = 0

        self._summaries = None
        self.global_step = None

    def build_graph(self):
        self._add_placeholder()
        self._add_seq2seq()
        self.global_step = tf.Variable(0, name='global_step', trainable=False)

        if self._hyper_params.mode == 'train':
            self._add_train_op()

        self._summaries = tf.summary.merge_all()

    def train(self, sess, config, article_batch, targets, article_lens, abstract_lens, loss_weights):
        to_return = [self._train_op, self._summaries, self._loss, self.global_step]

        return sess.run(
            to_return,
            config=config,
            feed_dict={
                self._articles: article_batch,
                self._targets: targets,
                self._article_lens: article_lens,
                self._abstract_lens: abstract_lens,
                self._loss_weights: loss_weights
            }
        )

    def eval(self, sess, config, article_batch, abstract_batch, targets, article_lens, abstract_lens, loss_weights):
        to_return = [self._summaries, self._loss, self.global_step]
        return sess.run(
            to_return,
            config=config,
            feed_dict={
                self._articles: article_batch,
                self._abstracts: abstract_batch,
                self._targets: targets,
                self._article_lens: article_lens,
                self._abstract_lens: abstract_lens,
                self._loss_weights: loss_weights
            }
        )

    def decode(self, sess, config, article_batch, abstract_batch, targets, article_lens, abstract_lens, loss_weights):
        to_return = [self._outputs, self.global_step]
        return sess.run(
            to_return,
            config=config,
            feed_dict={
                self._articles: article_batch,
                self._abstracts: abstract_batch,
                self._targets: targets,
                self._article_lens: article_lens,
                self._abstract_lens: abstract_lens,
                self._loss_weights: loss_weights
            }
        )

    def _next_device(self):
        if self._num_gpus == 0:
            return ''

        dev = '/gpu:%d' % self._cur_gpu
        if self._num_gpus > 1:
            self._cur_gpu = (self._cur_gpu + 1) % (self._num_gpus - 1)

        return dev

    def _get_gpu(self, gpu_id):
        if self._num_gpus <= 0 or gpu_id >= self._num_gpus:
            return ''

        return '/gpu:%d' % gpu_id

    def _add_placeholder(self):
        """Inputs to be fed into the graph"""
        hyper_params = self._hyper_params
        batch_size = hyper_params.batch_size
        enc_timesteps = hyper_params.enc_timesteps
        dec_timesteps = hyper_params.dec_timesteps

        self._articles = tf.placeholder(tf.int32, [batch_size, enc_timesteps], name='articles')
        self._abstracts = tf.placeholder(tf.int32, [batch_size, dec_timesteps], name='abstracts')
        self._targets = tf.placeholder(tf.int32, [batch_size, dec_timesteps], name='targets')
        self._article_lens = tf.placeholder(tf.int32, [batch_size], name='article_lens')
        self._abstract_lens = tf.placeholder(tf.int32, [batch_size], name='abstract_lens')
        self._loss_weights = tf.placeholder(tf.float32, [batch_size, dec_timesteps], name='loss_weights')

    def _add_seq2seq(self):
        vocab_size = self._vocab.num_ids()

        hyper_params = self._hyper_params
        embedding_size = hyper_params.emb_dim
        enc_layers = hyper_params.enc_layers

        with tf.variable_scope('seq2seq'):
            encoder_inputs = tf.unstack(tf.transpose(self._articles))
            decoder_inputs = tf.unstack(tf.transpose(self._abstracts))
            targets = tf.unstack(tf.transpose(self._targets))
            loss_weights = tf.unstack(tf.transpose(self._loss_weights))

            article_lens = self._article_lens

            # TODO: initialize using pre-trained embedding
            with tf.variable_scope('embedding'), tf.device('/cpu:0'):
                embedding = tf.get_variable(
                    'embedding', [vocab_size, embedding_size], dtype=tf.float32,
                    initializer=tf.random_normal_initializer(stddev=1e-4)
                )

                emb_encoder_inputs = [tf.nn.embedding_lookup(embedding, x) for x in encoder_inputs]
                emb_decoder_inputs = [tf.nn.embedding_lookup(embedding, x) for x in decoder_inputs]

            for layer_i in xrange(enc_layers):
                with tf.variable_scope('encoder%d' % layer_i), tf.device(self._next_device()):
                    cell_fw = tf.contrib.rnn.LSTMCell(
                        hyper_params.num_hidden,
                        initializer=tf.contrib.layers.xavier_initializer(),
                        state_is_tuple=False
                    )

                    cell_bw = tf.contrib.rnn.LSTMCell(
                        hyper_params.num_hidden,
                        initializer=tf.contrib.layers.xavier_initializer(),
                        state_is_tuple=False
                    )

                    (emb_encoder_inputs, fw_state, _) = tf.contrib.rnn.static_bidirectional_rnn(
                        cell_fw, cell_bw, emb_encoder_inputs, dtype=tf.float32, sequence_length=article_lens
                    )

            encoder_outputs = emb_encoder_inputs

            with tf.variable_scope('output_projection'):
                w = tf.get_variable(
                    'w', [hyper_params.num_hidden, vocab_size], dtype=tf.float32,
                    initializer=tf.truncated_normal_initializer(stddev=1e-4)
                )

                w_t = tf.transpose(w)
                v = tf.get_variable(
                    'v', [vocab_size], dtype=tf.float32,
                    initializer=tf.truncated_normal_initializer(stddev=1e-4)
                )

            with tf.variable_scope('decoder'), tf.device(self._next_device()):
                # when decoding, use model output from the previous step for the next step
                loop_function = None
                if hyper_params.mode == 'decode':
                    loop_function = self._extract_argmax_and_embed(embedding, (w, v), update_embedding=False)

                cell = tf.contrib.rnn.LSTMCell(
                    hyper_params.num_hidden,
                    initializer=tf.contrib.layers.xavier_initializer(),
                    state_is_tuple=False
                )

                encoder_outputs = [
                    tf.reshape(x, [hyper_params.batch_size, 1, 2 * hyper_params.num_hidden]) for x in encoder_outputs
                ]

                self._enc_top_states = tf.concat(axis=1, values=encoder_outputs)
                self._dec_in_state = fw_state

                # During decoding, follow up _dec_in_state are fed from beam_search, dec_out_state are stored by
                # beam_search for next feeding.
                initial_state_attention = (hyper_params.mode == 'decode')
                decoder_outputs, self._dec_out_state = tf.contrib.legacy_seq2seq.attention_decoder(
                    emb_decoder_inputs, self._dec_in_state, self._enc_top_states,
                    cell, num_heads=1, loop_function=loop_function,
                    initial_state_attention=initial_state_attention
                )

            with tf.variable_scope('output'), tf.device(self._next_device()):
                model_outputs = []
                for i in xrange(len(decoder_outputs)):
                    if i > 0:
                        tf.get_variable_scope().reuse_variables()

                    model_outputs.append(tf.nn.xw_plus_b(decoder_outputs[i], w, v))

            if hyper_params.mode == 'decode':
                with tf.variable_scope('decode_output'), tf.device('/cpu:0'):
                    best_outputs = [tf.arg_max(x, 1) for x in model_outputs]

                    tf.logging.info('best_outputs%s', best_outputs[0].get_shape())

                    self._outputs = tf.concat(
                        axis=1, values=[tf.reshape(x, [hyper_params.batch_size, 1]) for x in best_outputs]
                    )

                    self._topk_log_probs, self._topk_ids = tf.nn.top_k(
                        tf.log(tf.nn.softmax(model_outputs[-1])), hyper_params.batch_size * 2
                    )

            with tf.variable_scope('loss'), tf.device(self._next_device()):
                def sample_loss_func(inputs, labels):
                    with tf.device('/cpu:0'):  # TODO: Try gpu
                        labels = tf.reshape(labels, [-1, 1])
                        return tf.nn.sampled_softmax_loss(
                            weights=w_t, biases=v, labels=labels, inputs=inputs,
                            num_sampled=hyper_params.num_softmax_samples, num_classes=vocab_size
                        )

                if hyper_params.num_softmax_samples != 0 and hyper_params.mode == 'train':
                    self._loss = seq2seq_lib.sampled_sequence_loss(
                        decoder_outputs, targets, loss_weights, sample_loss_func
                    )
                else:
                    self._loss = tf.contrib.legacy_seq2seq.sequence_loss(
                        model_outputs, targets, loss_weights
                    )

                tf.summary.scalar('loss', tf.minimum(12.0, self._loss))

    def _add_train_op(self):
        hyper_params = self._hyper_params

        self._lr = tf.maximum(
            hyper_params.min_lr,
            tf.train.exponential_decay(hyper_params.lr, self.global_step, 30000, 0.98)
        )

        t_vars = tf.trainable_variables()
        with tf.device(self._get_gpu(self._num_gpus - 1)):
            grads, global_norm = tf.clip_by_global_norm(tf.gradients(self._loss, t_vars), hyper_params.max_grad_norm)

        tf.summary.scalar('global_norm', global_norm)

        optimizer = tf.train.AdagradOptimizer(self._lr)
        tf.summary.scalar('learning rate', self._lr)

        self._train_op = optimizer.apply_gradients(zip(grads, t_vars), global_step=self.global_step, name='train_step')

    def encode_top_state(self, sess, enc_inputs, enc_len):
        """get the top states from encoder from decoder.

        Args:
            sess: tensorflow session.
            enc_inputs: encoder inputs of shape [batch_size, enc_timesteps].
            enc_len: encoder input length of shape [batch_size]

        Returns:
            enc_top_states: the top level encoder stats.
            dec_in_state: the decoder layer initial state.
        """
        results = sess.run(
            [self._enc_top_states, self._dec_in_state],
            feed_dict={self._articles: enc_inputs, self._article_lens: enc_len}
        )

        return results[0], results[1][0]

    def decode_top_k(self, sess, latest_tokens, enc_top_states, dec_init_states):
        results = sess.run(
            [self._topk_ids, self._topk_log_probs, self._dec_out_state],
            feed_dict={
                self._enc_top_states: enc_top_states,
                self._dec_in_state: np.squeeze(np.array(dec_init_states)),
                self._abstracts: np.transpose(np.array([latest_tokens])),
                self._abstract_lens: np.ones([len(dec_init_states)], np.int32)
            }
        )

        ids, probs, states = results[0], results[1], results[2]
        new_states = [s for s in states]

        return ids, probs, new_states

    @staticmethod
    def _extract_argmax_and_embed(embedding, output_projection=None, update_embedding=True):
        """Get a loop function that extracts the previous symbol and embeds it.

        Args:
            embedding: embedding tensor for symbols.
            output_projection: None or a pair (W, B). If provided, each fed previous output will first be multiplied
                by W and added B.
            update_embedding: Boolean, if False, the gradients will not propagate through the embeddings.

        Returns:
            A loop function.
        """

        def loop_function(prev, _):
            """function that feed previous model output rather than ground truth"""
            if output_projection is not None:
                prev = tf.nn.xw_plus_b(prev, output_projection[0], output_projection[1])

            prev_symbol = tf.arg_max(prev, 1)
            emb_prev = tf.nn.embedding_lookup(embedding, prev_symbol)
            if not update_embedding:
                emb_prev = tf.stop_gradient(emb_prev)

            return emb_prev

        return loop_function
