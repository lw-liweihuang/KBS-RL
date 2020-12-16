"""
Created on Fri Jul  6 18:36:38 2018

@author: fms
"""

import tensorflow as tf
from tensorflow.contrib.seq2seq import *
from tensorflow.python.layers.core import Dense
from tensorflow.python.ops import math_ops
from tensorflow.python.framework import ops
from tensorflow.python.util import nest
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import rnn

from tensorflow.python.ops import rnn_cell_impl
import numpy as np
import collections


class BasicDecoderHit(collections.namedtuple("BasicDecoderHit", "hit")):
    pass


class SimpleEmbeddingWrapper(tf.contrib.rnn.RNNCell):

    def __init__(self, cell, embeddings, reuse=None):
        super(SimpleEmbeddingWrapper, self).__init__(_reuse=reuse)
        # rnn_cell_impl.assert_like_rnncell("cell", cell)
        self._cell = cell
        self.embeddings = embeddings

    @property
    def state_size(self):
        return self._cell.state_size

    @property
    def output_size(self):
        return self._cell.output_size

    def zero_state(self, batch_size, dtype):
        with ops.name_scope(type(self).__name__ + "ZeroState", values=[batch_size]):
            return self._cell.zero_state(batch_size, dtype)

    def call(self, inputs, state):
        with ops.device("/cpu:0"):
            if isinstance(state, tuple):
                data_type = state[0].dtype
            else:
                data_type = state.dtype

            embedded = tf.nn.embedding_lookup(self.embeddings, array_ops.reshape(inputs, [-1]))

            return self._cell(embedded, state)


class InteractiveGreedyEmbeddingHelper(Helper):

    def __init__(self, embedding, k, start_tokens, start_hit, size, sequence_length):
        
        if callable(embedding):
            self._embedding_fn = embedding
        else:
            self._embedding_fn = (lambda ids: tf.nn.embedding_lookup(embedding, ids))

        self._start_tokens = tf.convert_to_tensor(start_tokens, dtype=tf.int32, name="start_tokens")
        self.start_hit = start_hit

        if self._start_tokens.get_shape().ndims != 1:
            raise ValueError("start_tokens must be a vector")

        self._batch_size = tf.size(start_tokens)
        self._start_inputs = self._embedding_fn(self._start_tokens) * self.start_hit
        self._k = k
        self._output_size = size
        self._sequence_length = sequence_length

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def sample_ids_shape(self):
        return tf.TensorShape([])

    @property
    def sample_ids_dtype(self):
        return tf.int32

    def initialize(self, name=None):
        finished = tf.tile([False], [self._batch_size])
        return finished, self._start_inputs

    def sample(self, time, outputs, state, name=None):
        del time, state
        if not isinstance(outputs, ops.Tensor):
            raise TypeError("Expected outputs to be a single Tensor, got: %s" % type(outputs))

        sample_ids = tf.argmax(outputs, axis=-1, output_type=tf.int32)
        return sample_ids

    def next_inputs(self, time, outputs, state, history_masking, interesting, name=None):

        next_time = time + 1
        finished = (next_time >= self._sequence_length)
        all_finished = math_ops.reduce_all(finished)
        masking_outputs = outputs * history_masking
        top_n_index = tf.nn.top_k(masking_outputs, self._k).indices
        one_hot_top_n_index = tf.reduce_sum(tf.one_hot(top_n_index, depth=self._output_size, axis=-1), axis=-2)

        current_interesting_index = interesting * one_hot_top_n_index
        hit_flag = tf.reduce_sum(current_interesting_index, axis=-1) > 0
        hit_flag_factor = tf.reshape(tf.cast(hit_flag, tf.float32), (-1, 1))
        current_interesting_score = hit_flag_factor * current_interesting_index * outputs + (
                1 - hit_flag_factor) * masking_outputs

        selected_item = tf.argmax(current_interesting_score, axis=-1)
        one_hot_selected_item = tf.one_hot(selected_item, depth=self._output_size, axis=-1)
                    
        emb = self._embedding_fn(selected_item)
        output_emb = hit_flag_factor * emb + (hit_flag_factor - 1) * emb

        next_history_masking = history_masking - hit_flag_factor * one_hot_selected_item

        next_inputs = tf.cond(all_finished, lambda: self._start_inputs, lambda: output_emb)

        return (finished, next_inputs, state, selected_item, next_history_masking, interesting, hit_flag_factor,
                current_interesting_index)


class InteractiveDecoder(BasicDecoder):

    def __init__(self, cell, helper, initial_state, initial_history_masking, interesting, output_layer=None):
        self._cell = cell
        self._helper = helper
        self._initial_state = initial_state
        self._output_layer = output_layer
        self._initial_history_masking = initial_history_masking
        self._interesting = interesting

    @property
    def hit_size(self):
        return BasicDecoderHit(hit=tf.TensorShape([]))

    @property
    def hit_dtype(self):
        return BasicDecoderHit(tf.float32)

    def initialize(self, name=None):
        return self._helper.initialize() + (self._initial_state,) + (self._initial_history_masking,) + (self._interesting,)

    def step(self, time, inputs, state, history, name=None):
        with tf.name_scope(name, "InteractiveDecoderStep", (time, inputs, state)):
            cell_outputs, cell_state = self._cell(inputs, state)
            if self._output_layer is not None:
                cell_outputs = self._output_layer(cell_outputs)

            (finished, next_inputs, next_state, sample_ids, next_history_masking, next_interesting,
             hit_flag_factor, current_interesting_index) = self._helper.next_inputs(
                time=time,
                outputs=cell_outputs,
                state=cell_state,
                history_masking=history,
                interesting=self._interesting)

        sample_ids = tf.cast(sample_ids, tf.int32)

        outputs = BasicDecoderOutput(cell_outputs, sample_ids)

        hit = BasicDecoderHit(hit_flag_factor)
        return (outputs, next_state, next_inputs, next_history_masking, hit, finished)


class ExternalMemInteractiveDecoder(BasicDecoder):

    def __init__(self, cell, helper, initial_state, initial_history_masking, interesting, mem, rnn_size,
                 output_layer=None):
        self._cell = cell
        self._helper = helper
        self._initial_state = initial_state
        self._output_layer = output_layer
        self._initial_history_masking = initial_history_masking
        self._interesting = interesting
        
        self.mem_output_layer = tf.layers.Dense(rnn_size, use_bias=False, activation=tf.nn.tanh)
        self._mem = self.mem_output_layer(mem)
        self.forget_gate_mem_part = tf.layers.Dense(rnn_size, use_bias=True, activation=tf.nn.relu)
        self.forget_gate_cell_part = tf.layers.Dense(rnn_size, use_bias=False, activation=tf.nn.relu)

    @property
    def hit_size(self):
        return BasicDecoderHit(hit=tf.TensorShape([]))

    @property
    def hit_dtype(self):
        return BasicDecoderHit(tf.float32)

    def initialize(self, name=None):
        return self._helper.initialize() + (self._initial_state,) + (self._initial_history_masking,) + (
            self._interesting,)

    def step(self, time, inputs, state, history, name=None):
        with tf.name_scope(name, "InteractiveDecoderStep", (time, inputs, state)):
            cell_outputs, cell_state = self._cell(inputs, state)
            
            mem_gate = tf.nn.sigmoid(-self.forget_gate_mem_part(self._mem) + self.forget_gate_cell_part(cell_outputs))
          
            cell_outputs = mem_gate * self._mem + (1 - mem_gate) * cell_outputs

            
            if self._output_layer is not None:
                cell_outputs = self._output_layer(cell_outputs)

            (finished, next_inputs, next_state, sample_ids, next_history_masking, next_interesting, hit_flag_factor,
             current_interesting_index) = self._helper.next_inputs(time=time, outputs=cell_outputs,
                                                                   state=cell_state,
                                                                   history_masking=history,
                                                                   interesting=self._interesting)

        sample_ids = tf.cast(sample_ids, tf.int32)
        outputs = BasicDecoderOutput(cell_outputs, sample_ids)
        hit = BasicDecoderHit(hit_flag_factor)
        return outputs, next_state, next_inputs, next_history_masking, hit, finished




def _create_zero_outputs(size, dtype, batch_size):
    def _create(s, d):
        return rnn_cell_impl._zero_state_tensors(s, batch_size, d)

    return nest.map_structure(_create, size, dtype)


def dynamic_interactive_decode(decoder, output_time_major=False, impute_finished=False, maximum_iterations=None,
                               parallel_iterations=32, swap_memory=False, scope=None):
    with tf.variable_scope(scope, "decoder") as varscope:
        if varscope.caching_device is None:
            varscope.set_caching_device(lambda op: op.device)

        if maximum_iterations is not None:
            maximum_iterations = tf.convert_to_tensor(maximum_iterations, dtype=tf.int32, name="maximum_iterations")
            if maximum_iterations.get_shape().ndims != 0:
                raise ValueError("maximum_iterations must be a scalar")

        initial_finished, initial_inputs, initial_state, initial_history_masking, initial_interesting = decoder.initialize()

        zero_outputs = _create_zero_outputs(decoder.output_size,
                                            decoder.output_dtype,
                                            decoder.batch_size)

        if maximum_iterations is not None:
            initial_finished = tf.logical_or(
                initial_finished, 0 >= maximum_iterations)

        initial_sequence_lengths = tf.zeros_like(
            initial_finished, dtype=tf.int32)

        initial_time = tf.constant(0, dtype=tf.int32)

        def _shape(batch_size, from_shape):
            if (not isinstance(from_shape, tf.TensorShape) or from_shape.ndims == 0):
                return tf.TensorShape(None)
            else:
                batch_size = tensor_util.constant_value(ops.convert_to_tensor(batch_size, name="batch_size"))

                return tf.TensorShape([batch_size]).concatenate(from_shape)

        def _create_ta(s, d):
            return tf.TensorArray(
                dtype=d,
                size=0,
                dynamic_size=True,
                element_shape=_shape(decoder.batch_size, s))

        initial_outputs_ta = nest.map_structure(_create_ta, decoder.output_size, decoder.output_dtype)

        initial_hit_ta = nest.map_structure(_create_ta, decoder.hit_size, decoder.hit_dtype)

        def condition(unused_time, unused_outputs_ta, unused_state, unused_inputs, unused_history_masking, unused_hit,
                      finished, unused_sequence_lengths):
            return math_ops.logical_not(math_ops.reduce_all(finished))

        def body(time, outputs_ta, state, inputs, history_masking, hit_ta, finished, sequence_lengths):

            (next_outputs, decoder_state, next_inputs, next_history_masking, next_hit, decoder_finished) = \
                decoder.step(time, inputs, state, history_masking)

            if decoder.tracks_own_finished:
                next_finished = decoder_finished
            else:
                next_finished = math_ops.logical_or(decoder_finished, finished)

            if maximum_iterations is not None:
                next_finished = math_ops.logical_or(next_finished, time + 1 >= maximum_iterations)

            next_sequence_lengths = array_ops.where(math_ops.logical_and(math_ops.logical_not(finished), next_finished),
                                                    array_ops.fill(array_ops.shape(sequence_lengths), time + 1),
                                                    sequence_lengths)

            nest.assert_same_structure(state, decoder_state)
            nest.assert_same_structure(outputs_ta, next_outputs)
            nest.assert_same_structure(inputs, next_inputs)
            nest.assert_same_structure(history_masking, next_history_masking)

            nest.assert_same_structure(hit_ta, next_hit)

            if impute_finished:
                emit = nest.map_structure(lambda out, zero: array_ops.where(finished, zero, out),
                                          next_outputs, zero_outputs)
            else:
                emit = next_outputs

            def _maybe_copy_state(new, cur):
                if isinstance(cur, tensor_array_ops.TensorArray):
                    pass_through = True
                else:
                    new.set_shape(cur.shape)
                    pass_through = (new.shape.ndims == 0)
                return new if pass_through else array_ops.where(finished, cur, new)

            if impute_finished:
                next_state = nest.map_structure(
                    _maybe_copy_state, decoder_state, state)
            else:
                next_state = decoder_state

            outputs_ta = nest.map_structure(lambda ta, out: ta.write(time, out), outputs_ta, emit)

            hit_ta = nest.map_structure(lambda ta, out: ta.write(time, out), hit_ta, next_hit)

            return (time + 1, outputs_ta, next_state, next_inputs, next_history_masking, hit_ta, next_finished,
                    next_sequence_lengths)

        res = tf.while_loop(condition, body, loop_vars=[initial_time, initial_outputs_ta, initial_state,
                                                        initial_inputs, initial_history_masking,
                                                        initial_hit_ta, initial_finished, initial_sequence_lengths, ],
                            parallel_iterations=parallel_iterations, swap_memory=swap_memory)

        final_outputs_ta = res[1]
        final_state = res[2]
        final_history_masking = res[4]
        final_hit_ta = res[5]
        final_sequence_lengths = res[-1]

        final_outputs = nest.map_structure(lambda ta: ta.stack(), final_outputs_ta)

        final_hit = nest.map_structure(lambda ta: ta.stack(), final_hit_ta)
        try:
            final_outputs, final_state = decoder.finalize(final_outputs, final_state, final_sequence_lengths)
        except NotImplementedError:
            pass

        if not output_time_major:
            final_outputs = nest.map_structure(rnn._transpose_batch_time, final_outputs)
            final_hit = nest.map_structure(rnn._transpose_batch_time, final_hit)

    return final_outputs, final_state, final_history_masking, final_hit, final_sequence_lengths


class InteractiveModel(object):

    def __init__(self, sess, rnn_size, layer_size,
                 decoder_vocab_size, embedding_dim, k, lr):

        self.sess = sess
        self._k = k
        self.lr = lr
        self.postive_imediate_reward = 1.0

        self.negative_imediate_reward = 0.2
        self.account_ratio = 0.9

        self.rnn_size = rnn_size

        self.interesting = tf.placeholder(tf.float32, shape=[None, decoder_vocab_size], name='interest')
        self.history_masking = tf.placeholder(tf.float32, shape=[None, decoder_vocab_size], name='history')

        decoder_cell = self._get_simple_lstm(rnn_size, layer_size)

        self.rnn_init_state = tuple([tf.placeholder(tf.float32, [1, rnn_size],
                                                    name='rnn_state') for _ in range(layer_size)])




        decoder_embedding = tf.Variable(tf.truncated_normal(shape=[decoder_vocab_size, embedding_dim], stddev=0.1),
                                            name='decoder_embedding')


        self.encoder_items = tf.placeholder(tf.int32, shape=[None, None, 1], name='memory_encoder')

        self.encoder_sequence_length = tf.placeholder(tf.int32, shape=[None], name='memory_seq_length')

        encoder_cell = SimpleEmbeddingWrapper(decoder_cell, decoder_embedding)

        encoder_outputs, self.encoder_final_state = tf.nn.dynamic_rnn(cell=encoder_cell,
                                                                      inputs=self.encoder_items,
                                                                      sequence_length=self.encoder_sequence_length,
                                                                      dtype=tf.float32)

        self.start_tokens = tf.placeholder(tf.int32, shape=[None], name='start_tokens')

        self.start_hit = tf.placeholder(tf.float32, shape=[None], name='start_hit')

        self.sequence_length = tf.placeholder(tf.int32, shape=[None], name='seq_length')

        helper = InteractiveGreedyEmbeddingHelper(decoder_embedding, self._k, self.start_tokens, self.start_hit,
                                                  decoder_vocab_size, self.sequence_length)

        with tf.variable_scope('decoder'):
            fc_layer = Dense(decoder_vocab_size, activation=tf.nn.softmax)

            decoder = InteractiveDecoder(decoder_cell, helper, self.rnn_init_state,
                                         self.history_masking, self.interesting, fc_layer)

        self.logits, self.final_state, self.final_history_masking, self.hit, self.final_sequence_lengths = \
            dynamic_interactive_decode(decoder)

        self.hit = self.hit.hit

        reverse_hit = tf.reverse_sequence(self.hit, self.sequence_length, seq_dim=1)

        self.reverse_imediate_reward = tf.where(reverse_hit > 0,
                                                reverse_hit * self.postive_imediate_reward,
                                                (reverse_hit - 1) * self.negative_imediate_reward)

        self.imediate_reward = tf.reverse_sequence(self.reverse_imediate_reward, self.sequence_length, seq_dim=1)

        initial_time = tf.constant(0, dtype=tf.int32)

        initial_pre_reward = self.reverse_imediate_reward[0, 0] * 0.0

        output_ta = tf.TensorArray(dtype=tf.float32, size=1, dynamic_size=True)

        def cond(time, pre_reward, output_ta_l):
            return tf.reduce_all(time < self.sequence_length)

        def body(time, pre_reward, output_ta_l):
            pre_reward = self.reverse_imediate_reward[0, time] + self.account_ratio * pre_reward

            output_ta_l = output_ta_l.write(time, pre_reward)
            return time + 1, pre_reward, output_ta_l

        res = tf.while_loop(cond, body, loop_vars=[initial_time, initial_pre_reward, output_ta])

        self.cumsum_reward = tf.reverse_sequence([res[-1].stack()], self.sequence_length, seq_dim=1)
        self.cumsum_reward = tf.stop_gradient(self.cumsum_reward)
        self.rnn_output = self.logits.rnn_output
        self.sample_ids = self.logits.sample_id
        self.onehot_sample = tf.one_hot(self.sample_ids, depth=decoder_vocab_size, axis=-1)
        self.target = tf.placeholder(tf.int32, shape=[None, None], name='target')
        self.onehot_target = tf.one_hot(self.target, depth=decoder_vocab_size, axis=-1)

        
        self.gt_ratio = tf.cumprod((self.cumsum_reward * 0 + 1) * self.account_ratio, axis=1)

        self.gt_ratio = tf.stop_gradient(self.gt_ratio)
        self.is_reinforce = tf.placeholder(tf.int32, shape=[], name='isReinfoce')


        self.reinforce_cross_entropy = tf.reduce_mean(-tf.reduce_sum(tf.log(1e-8 + tf.reshape(self.rnn_output,
                                                                                       [-1,
                                                                                        decoder_vocab_size])) * tf.reshape(
            self.onehot_sample, [-1, decoder_vocab_size]) * self.cumsum_reward * self.gt_ratio,
                                                                     axis=-1), name='reinfolearn')

        self.supervised_cross_entropy = tf.reduce_mean(-tf.reduce_sum(
            tf.log(1e-8 + tf.reshape(self.rnn_output, [-1, decoder_vocab_size])) *
            tf.reshape(self.onehot_target, [-1, decoder_vocab_size]), name='mem_suplearn'))

        self.cost = tf.cond(self.is_reinforce > 0, lambda: self.reinforce_cross_entropy,
                            lambda: self.supervised_cross_entropy)

        self.train_opt = tf.train.AdamOptimizer(self.lr, epsilon=1e-4)

        gradients = self.train_opt.compute_gradients(self.cost)
        capped_gradients = [(tf.clip_by_value(grad, -10., 10.), var) for grad, var in gradients if grad is not None]
        self.train_op = self.train_opt.apply_gradients(capped_gradients)

        # self.train_op = self.train_opt.minimize(self.cost)

    def inference_warm_state(self, known_items):
        encoder_sequence_length = np.asarray([len(known_items), ])
        encoder_items = np.asarray(known_items).reshape((1, -1, 1))
        feed_dict = {self.encoder_items: encoder_items, self.encoder_sequence_length: encoder_sequence_length}

        return self.sess.run(self.encoder_final_state, feed_dict=feed_dict)

    def reinforcement_learn(self, user_interesting, user_masking, processes_length, init_state=None, s_token=0,
                            s_hit=1.0):

        s_token = np.asarray([s_token, ], dtype='int32')
        s_hit = np.asarray([s_hit, ], dtype='float32')
        fake_target = np.zeros((1, processes_length), dtype='int32')
        processes_length = np.asarray([processes_length, ])

        feed_dict = {self.interesting: user_interesting,
                     self.history_masking: user_masking,
                     self.start_tokens: s_token,
                     self.start_hit: s_hit,
                     self.sequence_length: processes_length,
                     self.target: fake_target,
                     self.is_reinforce: 1}

        if init_state is None:
            for i in self.rnn_init_state:
                feed_dict[i] = np.zeros([1, self.rnn_size], dtype='float32')
        else:
            for i, j in zip(self.rnn_init_state, init_state):
                feed_dict[i] = j

        return self.sess.run([self.train_op, self.hit, self.final_state, self.final_history_masking,
                              self.sample_ids], feed_dict=feed_dict)

    def supervised_learn(self, user_interesting, user_masking, targets, processes_length,
                         init_state=None, s_token=0, s_hit=1.0):

        s_token = np.asarray([s_token, ], dtype='int32')
        s_hit = np.asarray([s_hit, ], dtype='float32')

        targets = np.asarray(targets[:processes_length], dtype='int32').reshape((1, -1))
        processes_length = np.asarray([processes_length, ])

        feed_dict = {self.interesting: user_interesting,
                     self.history_masking: user_masking,
                     self.start_tokens: s_token,
                     self.start_hit: s_hit,
                     self.sequence_length: processes_length,
                     self.target: targets,
                     self.is_reinforce: 0}

        if init_state is None:
            for i in self.rnn_init_state:
                feed_dict[i] = np.zeros([1, self.rnn_size], dtype='float32')
        else:
            for i, j in zip(self.rnn_init_state, init_state):
                feed_dict[i] = j

        return self.sess.run([self.train_op, self.hit, self.final_state, self.final_history_masking,
                              self.sample_ids], feed_dict=feed_dict)

    def inference(self, user_interesting, user_masking, processes_length, init_state=None, s_t=0):

        s_token = np.asarray([s_t, ], dtype='int32')
        s_hit = np.asarray([1.0, ], dtype='float32')

        fake_target = np.zeros((1, processes_length), dtype='int32')
        processes_length = np.asarray([processes_length, ])

        feed_dict = {self.interesting: user_interesting,
                     self.history_masking: user_masking,
                     self.start_tokens: s_token,
                     self.start_hit: s_hit,
                     self.sequence_length: processes_length,
                     self.target: fake_target,
                     self.is_reinforce: 1}

        if init_state is None:
            for i in self.rnn_init_state:
                feed_dict[i] = np.zeros([1, self.rnn_size], dtype='float32')
        else:
            for i, j in zip(self.rnn_init_state, init_state):
                feed_dict[i] = j

        (user_item_probs, user_selected_items, user_final_masking,
         user_hit, user_imediate_reward, user_cumsum_reward) = self.sess.run(
            [self.rnn_output, self.sample_ids, self.final_history_masking, self.hit, self.imediate_reward,
             self.cumsum_reward], feed_dict=feed_dict)

        return (user_item_probs, user_selected_items, user_final_masking,
                user_hit, user_imediate_reward, user_cumsum_reward)

    def _get_simple_lstm(self, rnn_size, layer_size):
        lstm_layers = [tf.contrib.rnn.GRUCell(rnn_size) for _ in range(layer_size)]
        return tf.contrib.rnn.MultiRNNCell(lstm_layers)


class EMInteractiveModel(object):

    def __init__(self, sess, rnn_size, layer_size, decoder_vocab_size, embedding_dim, k, lr):

        self.sess = sess
        self._k = k
        self.lr = lr
        self.postive_imediate_reward = 1.0

        self.negative_imediate_reward = 0.2

        self.account_ratio = 0.9

        self.rnn_size = rnn_size
        
        
        self.interesting = tf.placeholder(tf.float32, shape=[None, decoder_vocab_size], name='interest')
        self.history_masking = tf.placeholder(tf.float32, shape=[None, decoder_vocab_size], name='history')

        decoder_cell = self._get_simple_lstm(rnn_size, layer_size)

        self.rnn_init_state = tf.placeholder(tf.float32, [1, rnn_size], name='rnn_state')


        decoder_embedding = tf.Variable(tf.truncated_normal(shape=[decoder_vocab_size, embedding_dim], stddev=0.1),
                                            name='decoder_embedding')


        self.start_tokens = tf.placeholder(tf.int32, shape=[None], name='start_tokens')

        self.start_hit = tf.placeholder(tf.float32, shape=[None], name='start_hit')

        self.mem = tf.placeholder(tf.float32, shape=[None, decoder_vocab_size], name='mem')

        self.sequence_length = tf.placeholder(tf.int32, shape=[None], name='seq_length')

        helper = InteractiveGreedyEmbeddingHelper(decoder_embedding, self._k, self.start_tokens, self.start_hit,
                                                  decoder_vocab_size, self.sequence_length)

        with tf.variable_scope('decoder'):
            fc_layer = Dense(decoder_vocab_size, activation=tf.nn.softmax)

            decoder = ExternalMemInteractiveDecoder(decoder_cell, helper, self.rnn_init_state,
                                                    self.history_masking, self.interesting, self.mem, self.rnn_size,
                                                    fc_layer)

        self.logits, self.final_state, self.final_history_masking, self.hit, self.final_sequence_lengths = \
            dynamic_interactive_decode(decoder)

        self.hit = self.hit.hit

        reverse_hit = tf.reverse_sequence(self.hit, self.sequence_length, seq_dim=1)

        self.reverse_imediate_reward = tf.where(reverse_hit > 0,
                                                reverse_hit * self.postive_imediate_reward,
                                                (reverse_hit - 1) * self.negative_imediate_reward)

        self.imediate_reward = tf.reverse_sequence(self.reverse_imediate_reward, self.sequence_length, seq_dim=1)

        initial_time = tf.constant(0, dtype=tf.int32)

        initial_pre_reward = self.reverse_imediate_reward[0, 0] * 0.0

        output_ta = tf.TensorArray(dtype=tf.float32, size=1, dynamic_size=True)

        def cond(time, pre_reward, output_ta_l):
            return tf.reduce_all(time < self.sequence_length)

        def body(time, pre_reward, output_ta_l):
            pre_reward = self.reverse_imediate_reward[0, time] + self.account_ratio * pre_reward

            output_ta_l = output_ta_l.write(time, pre_reward)
            return time + 1, pre_reward, output_ta_l

        res = tf.while_loop(cond, body, loop_vars=[initial_time, initial_pre_reward, output_ta])

        self.cumsum_reward = tf.reverse_sequence([res[-1].stack()], self.sequence_length, seq_dim=1)
        self.cumsum_reward = tf.stop_gradient(self.cumsum_reward)

        self.rnn_output = self.logits.rnn_output

        self.sample_ids = self.logits.sample_id

        self.onehot_sample = tf.one_hot(self.sample_ids, depth=decoder_vocab_size, axis=-1)

        self.target = tf.placeholder(tf.int32, shape=[None, None], name='target')

        self.onehot_target = tf.one_hot(self.target, depth=decoder_vocab_size, axis=-1)

        self.gt_ratio = tf.cumprod((self.cumsum_reward * 0 + 1) * self.account_ratio, axis=1)
        self.gt_ratio = tf.stop_gradient(self.gt_ratio)
        self.is_reinforce = tf.placeholder(tf.int32, shape=[], name='isReinfoce')

        self.reinforce_cross_entropy = tf.reduce_mean(-tf.reduce_sum(tf.log(1e-8 +  tf.reshape(self.rnn_output,
                                                                                       [-1, decoder_vocab_size])) *
                                                                     tf.reshape(self.onehot_sample, [-1, decoder_vocab_size]) *
                                                                     self.cumsum_reward * self.gt_ratio,
                                                                     axis=-1), name='reinfolearn')

        self.supervised_cross_entropy = tf.reduce_mean(-tf.reduce_sum(
            tf.log(1e-8 + tf.reshape(self.rnn_output, [-1, decoder_vocab_size])) *
            tf.reshape(self.onehot_target, [-1, decoder_vocab_size]), name='mem_suplearn'))

        self.cost = tf.cond(self.is_reinforce > 0, lambda: self.reinforce_cross_entropy,
                            lambda: self.supervised_cross_entropy)

        self.train_opt = tf.train.AdamOptimizer(self.lr, epsilon=1e-4)

        gradients = self.train_opt.compute_gradients(self.cost)
        capped_gradients = [(tf.clip_by_value(grad, -10., 10.), var) for grad, var in gradients if grad is not None]
        self.train_op = self.train_opt.apply_gradients(capped_gradients)
        # self.train_op = self.train_opt.minimize(self.cost)

    def reinforcement_learn(self, user_interesting, user_masking, mem, processes_length, init_state=None, s_token=0,
                            s_hit=1.0):

        s_token = np.asarray([s_token, ], dtype='int32')
        s_hit = np.asarray([s_hit, ], dtype='float32')
        fake_target = np.zeros((1, processes_length), dtype='int32')
        processes_length = np.asarray([processes_length, ])

        feed_dict = {self.interesting: user_interesting,
                     self.history_masking: user_masking,
                     self.mem: mem,
                     self.start_tokens: s_token,
                     self.start_hit: s_hit,
                     self.sequence_length: processes_length,
                     self.target: fake_target,
                     self.is_reinforce: 1}

        if init_state is None:
            feed_dict[self.rnn_init_state] = np.zeros([1, self.rnn_size], dtype='float32')
        else:
            feed_dict[self.rnn_init_state] = init_state

        return self.sess.run([self.cost, self.train_op, self.hit, self.final_state,
                              self.final_history_masking, self.sample_ids], feed_dict=feed_dict)

    def supervised_learn(self, user_interesting, user_masking, mem, targets, processes_length,
                         init_state=None, s_token=0, s_hit=1.0):

        s_token = np.asarray([s_token, ], dtype='int32')
        s_hit = np.asarray([s_hit, ], dtype='float32')

        targets = np.asarray(targets[:processes_length], dtype='int32').reshape((1, -1))

        processes_length = np.asarray([processes_length, ])

        feed_dict = {self.interesting: user_interesting,
                     self.history_masking: user_masking,
                     self.mem: mem,
                     self.start_tokens: s_token,
                     self.start_hit: s_hit,
                     self.sequence_length: processes_length,
                     self.target: targets,
                     self.is_reinforce: 0}

        if init_state is None:
            feed_dict[self.rnn_init_state] = np.zeros([1, self.rnn_size], dtype='float32')
        else:
            feed_dict[self.rnn_init_state] = init_state

        return self.sess.run([self.cost, self.train_op, self.hit, self.final_state, self.final_history_masking, self.sample_ids],
                             feed_dict=feed_dict)

    def inference(self, user_interesting, user_masking, mem, processes_length, init_state=None, s_t=0):
        s_token = np.asarray([s_t, ], dtype='int32')
        s_hit = np.asarray([1.0, ], dtype='float32')
        fake_target = np.zeros((1, processes_length), dtype='int32')
        processes_length = np.asarray([processes_length, ])
        feed_dict = {self.interesting: user_interesting,
                     self.history_masking: user_masking,
                     self.mem: mem,
                     self.start_tokens: s_token,
                     self.start_hit: s_hit,
                     self.sequence_length: processes_length,
                     self.target: fake_target,
                     self.is_reinforce: 1}

        if init_state is None:
            feed_dict[self.rnn_init_state] = np.zeros([1, self.rnn_size], dtype='float32')
        else:
            feed_dict[self.rnn_init_state] = init_state

        (user_item_probs, user_selected_items, user_final_masking,
         user_hit, user_imediate_reward, user_cumsum_reward) = self.sess.run(
            [self.rnn_output, self.sample_ids, self.final_history_masking, self.hit, self.imediate_reward,
             self.cumsum_reward], feed_dict=feed_dict)

        return (user_item_probs, user_selected_items, user_final_masking,
                user_hit, user_imediate_reward, user_cumsum_reward)

    def _get_simple_lstm(self, rnn_size, layer_size):
        lstm_layers = tf.contrib.rnn.GRUCell(rnn_size)
        return lstm_layers  # tf.contrib.rnn.MultiRNNCell(lstm_layers)




def get_cum_interesting(y, n):
    return np.asarray(np.sum(np.eye(n)[y], axis=0), dtype='float32').reshape([1, n])


def get_initial_masking(n):
    return np.ones((1, n), dtype='float32')


def get_masking(n, known_items):
    masking = get_initial_masking(n)
    masking -= np.asarray(np.sum(np.eye(n)[known_items], axis=0), dtype='float32').reshape([1, n])
    return masking


