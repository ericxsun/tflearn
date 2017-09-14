#!/user/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright @2017 Research, NTC Inc. (ntc.com)
#
# Author: Eric x.sun <eric.x.sun@gmail.com>
#

import tensorflow as tf

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_bool('normalize_by_length', True, 'whether to normalize or not')


class Hypothesis(object):
    def __init__(self, tokens, log_prob, state):
        """
        Args:
            tokens: start tokens for decoding.
            log_prob: log prob of the start tokens, usually 1.
            state: decoder initial states.
        """

        self.tokens = tokens
        self.log_prob = log_prob
        self.state = state

    def extend(self, token, log_prob, new_state):
        """extend the hypothesis with result from latest step.

        Args:
            token: latest token from decoding.
            log_prob: log prob of the latest decoded tokens.
            new_state: decoder output state, fed to the decoder for next step.

        Returns:
            new hypothesis with the results from latest step.
        """

        return Hypothesis(self.tokens + [token], self.log_prob + log_prob, new_state)

    @property
    def latest_token(self):
        return self.tokens[-1]

    def __str__(self):
        return 'Hypothesis(log prob=%.4f, tokens=%s)' % (self.log_prob, self.tokens)


class BeamSearch(object):
    def __init__(self, model, beam_size, start_token, end_token, max_steps):
        """
        Args:
            model: Seq2SeqAttentionModel.
            beam_size: int.
            start_token: int, id of the token to start decoding with.
            end_token: int, id of the token that completes an hypothesis
            max_steps: int, upper limit on the size of the hypothesis
        """
        self._model = model
        self._beam_size = beam_size
        self._start_token = start_token
        self._end_token = end_token
        self._max_steps = max_steps

    def beam_search(self, sess, enc_inputs, enc_seq_len):
        """performs beam search for decoding

        Args:
            sess: tf.Session
            enc_inputs: ndarray of shape [enc_length, 1], the document ids to encode.
            enc_seq_len: ndarray of shape [1], the length of the sequence.

        Returns:
            list of Hypothesis, the best hypotheses found by beam search, ordered by score.
        """

        enc_top_states, dec_in_state = self._model.encode_top_state(sess, enc_inputs, enc_seq_len)

        # replicate the initial states k times for the first step
        hyps = [Hypothesis([self._start_token], 0.0, dec_in_state)] * self._beam_size

        results = []
        steps = 0

        while steps < self._max_steps and len(results) < self._beam_size:
            latest_tokens = [h.latest_token for h in hyps]
            states = [h.state for h in hyps]

            topk_ids, topk_log_probs, new_states = self._model.decode_topk(
                sess, latest_tokens, enc_top_states, states
            )

            all_hyps = []

            # the first step takes the best k results from first hyps, following steps take the best k results from
            # k*k hyps.
            num_beam_source = 1 if steps == 0 else len(hyps)
            for i in xrange(num_beam_source):
                h, ns = hyps[i], new_states[i]
                for j in xrange(self._beam_size * 2):
                    all_hyps.append(h.extend(topk_ids[i, j], topk_log_probs[i, j], ns))

            # filter and collect any hypothesis that have the end token
            hyps = []
            for h in self._best_hyps(all_hyps):
                if h.latest_token == self._end_token:
                    results.append(h)  # Pull the hypothesis off the beam if the end token is reached.
                else:
                    hyps.append(h) # Otherwise continue to the extend the hypothesis.

                if len(hyps) == self._beam_size or len(results) == self._beam_size:
                    break

            steps += 1

        if steps == self._max_steps:
            results.extend(hyps)

        return self._best_hyps(results)

    @staticmethod
    def _best_hyps(hyps):
        if FLAGS.normalize_by_length:
            return sorted(hyps, key=lambda h: h.log_prob / len(h.tokens), reverse=True)
        else:
            return sorted(hyps, key=lambda h: h.log_prob, reverse=True)
