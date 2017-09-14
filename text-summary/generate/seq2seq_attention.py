#!/user/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright @2017 Research, NTC Inc. (ntc.com)
#
# Author: Eric x.sun <eric.x.sun@gmail.com>
#
import sys
import tensorflow as tf
import time

from seq2seq_attention_model import HYPER_PARAMS
from seq2seq_attention_model import Seq2SeqAttentionModel
from seq2seq_attention_decode import BeamSearchDecoder
import data
from batch_reader import BatchReader
from utils import LOGGER


FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('data_path', '', 'path pattern storing tf.Example')
tf.app.flags.DEFINE_string('vocab_path', '', 'filename of vocabulary file')
tf.app.flags.DEFINE_string('article_key', 'article', 'tf.Example feature key for article')
tf.app.flags.DEFINE_string('abstract_key', 'abstract', 'tf.Example feature key for abstract')
tf.app.flags.DEFINE_string('log_root', '', 'directory for log root.')
tf.app.flags.DEFINE_string('train_dir', '', 'directory for train')
tf.app.flags.DEFINE_string('eval_dir', '', 'directory for eval')
tf.app.flags.DEFINE_string('decode_dir', '', 'directory for decode summaries')

tf.app.flags.DEFINE_integer('vocab_max_size', 1000000, 'maximum number of words in vocabulary')
tf.app.flags.DEFINE_integer('max_iterations', 100000, 'maximum number of iterations')
tf.app.flags.DEFINE_integer('max_article_sentences', 10000, 'maximum number of first sentences to use from the article')
tf.app.flags.DEFINE_integer('max_abstract_sentences', 1000, 'maximum number of first sentences to use from the abstract')
tf.app.flags.DEFINE_integer('beam_size', 4, 'beam size for beam search decoding')
tf.app.flags.DEFINE_integer('eval_interval_secs', 60, 'frequency for running eval, in seconds')
tf.app.flags.DEFINE_integer('checkpoint_secs', 60, 'frequency for checkpoint, in seconds')
tf.app.flags.DEFINE_integer('num_gpus', 0, 'number of gpu used')
tf.app.flags.DEFINE_integer('random_seed', 111, 'a seed value for randomness')
tf.app.flags.DEFINE_integer('min_input_len', 2, 'minimum length of input, < this will be discard')

tf.app.flags.DEFINE_bool('use_bucketing', False, 'whether bucket articles for similar length')
tf.app.flags.DEFINE_bool('truncate_input', False, 'whether truncate inputs that are to long or not')

tf.app.flags.DEFINE_string('mode', '', 'train / eval / decode mode')
tf.app.flags.DEFINE_integer('batch_size', 4, 'batch size')
tf.app.flags.DEFINE_integer('num_hidden', 256, 'hidden units used in rnn-cell')
tf.app.flags.DEFINE_integer('emb_dim', 128, 'word embedding dimension')
tf.app.flags.DEFINE_integer('enc_layers', 4, 'encoder layers')
tf.app.flags.DEFINE_integer('enc_timesteps', 500, 'encode timesteps')
tf.app.flags.DEFINE_integer('dec_timesteps', 30, 'decode timesteps')
tf.app.flags.DEFINE_integer('max_grad_norm', 2, 'the maximum permissible norm of the gradient')
tf.app.flags.DEFINE_integer('num_softmax_samples', 4096, 'sampled softmax')
tf.app.flags.DEFINE_float('min_lr', 0.01, 'minimum learning rate')
tf.app.flags.DEFINE_float('lr', 0.15, 'learning rate')


def _running_avg_loss(loss, running_avg_loss, summary_writer, step, decay=0.999):
    """Calculate the running average of losses."""
    if running_avg_loss == 0:
        running_avg_loss = loss
    else:
        running_avg_loss = running_avg_loss * decay + (1 - decay) * loss

    running_avg_loss = min(running_avg_loss, 12)

    loss_sum = tf.Summary()
    loss_sum.value.add(tag='running_avg_loss', simple_value=running_avg_loss)
    summary_writer.add_summary(loss_sum, step)
    sys.stdout.write('running_avg_loss: %f\n' % running_avg_loss)

    return running_avg_loss


def _train(model, config, batch_reader):
    with tf.device('/cpu:0'):
        model.build_graph()

        saver = tf.train.Saver()
        summary_writer = tf.summary.FileWriter(FLAGS.train_dir)
        sv = tf.train.Supervisor(
            logdir=FLAGS.log_root,
            is_chief=True,
            saver=saver,
            summary_op=None,
            save_summaries_secs=60,
            save_model_secs=FLAGS.checkpoint_secs,
            global_step=model.global_step
        )

        sess = sv.prepare_or_wait_for_session(config=config)
        running_avg_loss = 0
        step = 0
        while not sv.should_stop() and step < FLAGS.max_iterations:
            (
                article_batch, abstract_batch, targets, article_lens, abstract_lens, loss_weights, _, _
            ) = batch_reader.next_batch()

            (
                _, summaries, loss, train_step
            ) = model.train(
                sess, config, article_batch, abstract_batch, targets, article_lens, abstract_lens, loss_weights
            )

            summary_writer.add_summary(summaries, train_step)
            running_avg_loss = _running_avg_loss(running_avg_loss, loss, summary_writer, train_step)
            step += 1
            if step % 100 == 0:
                summary_writer.flush()

        sv.stop()
        return running_avg_loss


def _eval(model, config, batch_reader, vocab=None):
    model.build_graph()

    saver = tf.train.Saver()
    summary_writer = tf.summary.FileWriter(FLAGS.eval_dir)
    sess = tf.Session(config=config)

    running_avg_loss = 0
    step = 0
    while True:
        time.sleep(FLAGS.eval_interval_secs)

        try:
            ckpt_state = tf.train.get_checkpoint_state(FLAGS.log_root)
        except tf.errors.OutOfRangeError as e:
            LOGGER.error('cannot restore checkpoint: %s', e)
            continue

        if not (ckpt_state and ckpt_state.model_checkpoint_path):
            LOGGER.info('no model to eval yet at %s', FLAGS.train_dir)
            continue

        LOGGER.info('loading checkpoint %s', ckpt_state.model_checkpoint_path)
        saver.restore(sess, ckpt_state.model_checkpoint_path)

        (
            article_batch, abstract_batch, targets, article_lens, abstract_lens, loss_weights, _, _
        ) = batch_reader.next_batch()

        (summaries, loss, train_step) = model.eval(
            sess, config, article_batch, abstract_batch, targets, article_lens, abstract_lens, loss_weights
        )

        LOGGER.info('article:  %s', ' '.join(data.convert_ids_to_words(article_batch[0][:].tolist(), vocab)))
        LOGGER.info('abstract: %s', ' '.join(data.convert_ids_to_words(abstract_batch[0][:].tolist(), vocab)))

        summary_writer.add_summary(summaries, train_step)
        running_avg_loss = _running_avg_loss(running_avg_loss, loss, summary_writer, train_step)
        if step % 100 == 0:
            summary_writer.flush()


def main(unused_argv):
    LOGGER.info('load vocab')
    vocab = data.Vocab(FLAGS.vocab_path, FLAGS.vocab_max_size)

    batch_size = FLAGS.batch_size
    if FLAGS.mode == 'decode':
        batch_size = FLAGS.beam_size

    hyper_params = HYPER_PARAMS(
        mode=FLAGS.mode,
        batch_size=batch_size,
        num_hidden=FLAGS.num_hidden,
        emb_dim=FLAGS.emb_dim,
        enc_layers=FLAGS.enc_layers,
        enc_timesteps=FLAGS.enc_timesteps,
        dec_timesteps=FLAGS.dec_timesteps,
        max_grad_norm=FLAGS.max_grad_norm,
        num_softmax_samples=FLAGS.num_softmax_samples,
        min_input_len=FLAGS.min_input_len,
        min_lr=FLAGS.min_lr,
        lr=FLAGS.lr
    )

    batch_reader = BatchReader(
        data_path=FLAGS.data_path,
        vocab=vocab,
        hyper_params=hyper_params,
        article_key=FLAGS.article_key,
        abstract_key=FLAGS.abstract_key,
        max_article_sentences=FLAGS.max_article_sentences,
        max_abstract_sentences=FLAGS.max_abstract_sentences,
        bucketing=FLAGS.use_bucketing,
        truncate_input=FLAGS.truncate_input
    )

    tf.set_random_seed(FLAGS.random_seed)

    config = tf.ConfigProto(
        gpu_options={"allow_growth": True},  # 按需增长
        device_count={"GPU": 2},  # limit to 2 GPU usage
        allow_soft_placement=True,
        inter_op_parallelism_threads=1,  # Nodes that perform blocking operations are enqueued on a pool
        intra_op_parallelism_threads=2  # The execution of an individual op (for some op types)
    )

    if FLAGS.mode == 'train':
        model = Seq2SeqAttentionModel(hyper_params, vocab, num_gpus=FLAGS.num_gpus)
        _train(model, config, batch_reader)
    elif FLAGS.mode == 'eval':
        model = Seq2SeqAttentionModel(hyper_params, vocab, num_gpus=FLAGS.num_gpus)
        _eval(model, config, batch_reader, vocab)
    elif FLAGS.mode == 'decode':
        decode_mdl_hps = hyper_params
        decode_mdl_hps.dec_timesteps = 1

        model = Seq2SeqAttentionModel(decode_mdl_hps, vocab, num_gpus=FLAGS.nu_gpus)
        decoder = BeamSearchDecoder(model, batch_reader, hyper_params, vocab)
        decoder.decode_loop()
    else:
        LOGGER.error('not supported mode: %s', hyper_params.mode)


if __name__ == '__main__':
    tf.app.run()
