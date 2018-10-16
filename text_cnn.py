# -*- coding: utf-8 -*-

import tensorflow as tf


class TextCnn():
    def __init__(self, config):
        self.sequence_length = config['sequence_length']
        self.num_classes = config['num_classes']
        self.vocab_size = config['vocab_size']
        self.embedding_size = config['embedding_size']
        self.filter_size = config['filter_size']
        self.num_filters = config['num_filters']
        self.l2_reg_lambda = config['l2_reg_lambda']
        self.device = config['device']

        # placeholder
        self.input_x = tf.placeholder(tf.int32, [None, self.sequence_length], name='input_x')
        self.input_y = tf.placeholder(tf.float32, [None, self.num_classes], name='input_y')
        self.dropout_keep_prob = tf.placeholder(tf.float32, name='dropout_keep_prob')

        # l2 loss
        l2_loss = tf.constant(0.0)

        # embedding layer
        with tf.device(self.device), tf.name_scope('embedding'):
            self.W = tf.Variable(
                tf.random_uniform([self.vocab_size, self.embedding_size], -1.0, 1.0),
                name='W'
            )
            self.embedded_chars = tf.nn.embedding_lookup(self.W, self.input_x)
            self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)

        # convolution+maxpooling layer
        pooled_outputs = []
        for i, filter_size in enumerate(self.filter_size):
            with tf.name_scope('conv-maxpool-{}'.format(i)):
                # convolution
                filter_shape = [self.filter_size[i], self.embedding_size, 1, self.num_filters]
                W = tf.Variable(
                    tf.truncated_normal(filter_shape, stddev=0.1), name='W'
                )
                b = tf.Variable(tf.constant(0.1, shape=[self.num_filters]), name='b')

                conv = tf.nn.conv2d(
                    self.embedded_chars_expanded,
                    W,
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name='conv'
                )

                # nonlinear
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name='relu')

                # max pooling
                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, self.sequence_length-self.filter_size[i]+1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name='pool'
                )
                pooled_outputs.append(pooled)

        # combine all pooled features
        num_filters_total = self.num_filters * len(self.filter_size)
        self.h_pool = tf.concat(pooled_outputs, 3)
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])

        # add dropout
        with tf.name_scope('dropout'):
            self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)

        # final scores and predictions
        with tf.name_scope('output'):
            W = tf.get_variable(
                "W",
                shape=[num_filters_total, self.num_classes],
                initializer=tf.contrib.layers.xavier_initializer()
            )
            b = tf.Variable(tf.constant(0.1, shape=[self.num_classes]), name='b')
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
            self.scores = tf.nn.xw_plus_b(self.h_drop, W, b, name='scores')
            self.predictions = tf.argmax(self.scores, 1, name='predictions')

        # loss
        with tf.name_scope('loss'):
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.input_y)
            self.loss = tf.reduce_mean(losses) + self.l2_reg_lambda*l2_loss

        # accuracy
        with tf.name_scope('accuracy'):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32), name='accuracy')
