#!/user/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright @2017 Research, NTC Inc. (ntc.com)
#
# Author: Eric x.sun <eric.x.sun@gmail.com>
#

import tensorflow as tf


def linear(args, output_size, bias, bias_start=0.0, scope=None):
    """Linear map: sum_i(args[i] * W[i]), where W[i] is a variable.

    Args:
        args: a 2D Tensor or a list of 2D Tensors.
        output_size: int, second dimension of W[i].
        bias: boolean, whether to add a bias or not.
        bias_start: starting value to initialize the bias, 0 by default.
        scope: VariableScope for the created sub-graph, 'Linear' by default.

    Return:
        A 2D Tensor with shape [batch, output_size], equal to
        sum_i(args[i] * W[i]), where W[i]s are newly created matrices.

    Raises:
        ValueError: if some of the arguments has unspecified or wrong shape.
    """
    if args is None or (isinstance(args, (list, tuple)) and not args):
        raise ValueError('`args` must be specified')

    if not isinstance(args, (list, tuple)):
        args = [args]

    total_arg_size = 0
    shapes = [a.get_shape().as_list() for a in args]
    for shape in shapes:
        if len(shape) != 2:
            raise ValueError('Linear is expecting 2D arguments: %s' % str(shapes))

        if not shape[1]:
            raise ValueError('Linear expects shape[1] of arguments: %s' % str(shapes))

        total_arg_size += shape[1]

    with tf.variable_scope(scope or 'Linear'):
        matrix = tf.get_variable('Matrix', [total_arg_size, output_size])

        res = tf.matmul(tf.concat(axis=1, values=args), matrix)

        if bias:
            bias_term = tf.get_variable('Bias', [output_size], initializer=tf.constant_initializer(bias_start))
            res += bias_term

    return res


def sequence_loss_by_example(inputs, targets, weights, loss_function, avg_across_timesteps=True, name=None):
    """sampled softmax loss for a sequence of inputs per example.

    Args:
        inputs: list of 2D Tensors of shape [batch_size, hidden_dimension].
        targets: list of 1D batch-sized int32 Tensors of the same length as inputs.
        weights: list of 1D batch-sized float Tensors of the same length as inputs.
        loss_function: sampled softmax function (inputs, labels) -> loss.
        avg_across_timesteps: if set, divide the returned cost by the total label weight.
        name: optional name for this operation, 'sampled_sequence_loss' by default.

    Returns:
        1D batch-sized float Tensor: the log-perplexity for each sequence.

    Raises:
        ValueError: if len(inputs) is different from len(targets) or len(weights).
    """

    len_t = len(targets)
    len_i = len(inputs)
    len_w = len(weights)

    if len_t != len_i or len_w != len_i:
        raise ValueError(
            'length of logits, weights, and targets must be the same %d, %d, %d' % (len_i, len_w, len_t)
        )

    with tf.name_scope(values=inputs + targets + weights, name=name, default_name='sequence_loss_by_example'):
        log_perplexity_list = []
        for _input, _target, _weight in zip(inputs, targets, weights):
            cross_loss = loss_function(_input, _target)
            log_perplexity_list.append(cross_loss * _weight)

        log_perplexities = tf.add_n(log_perplexity_list)
        if avg_across_timesteps:
            total_size = tf.add_n(weights)
            total_size += 1e-12  # just to avoid division by 0
            log_perplexities /= total_size

    return log_perplexities


def sampled_sequence_loss(
    inputs, targets, weights, loss_function, avg_across_timesteps=True, avg_across_batch=True, name=None
):
    """Weighted cross-entropy loss for sequence of logits, batch-collapsed.

    Args:
        inputs: list of 2D Tensors of shape [batch_size, hidden_dimension].
        targets: list of 1D batch-sized int32 Tensors of the same length as inputs.
        weights: list of 1D batch-sized float Tensors of the same length as inputs.
        loss_function: sampled softmax function (inputs, labels) -> loss.
        avg_across_timesteps: if set, divide the returned cost by the total label weight.
        avg_across_batch: if set, divide the returned cost by the batch size.
        name: optional name for this operation, 'sampled_sequence_loss' by default.

    Returns:
        A scalar float Tensor: the average log-perplexity per symbol (weighted).

    Raises:
        ValueError: if len(inputs) is different from len(targets) or len(weights).
    """
    with tf.name_scope(values=inputs + targets + weights, name=name, default_name='sampled_sequence_loss'):
        cost = tf.reduce_sum(sequence_loss_by_example(
            inputs, targets, weights, loss_function, avg_across_timesteps=avg_across_timesteps
        ))

        if avg_across_batch:
            batch_size = tf.shape(targets[0][0])
            return cost / tf.cast(batch_size, tf.float32)
        else:
            return cost
