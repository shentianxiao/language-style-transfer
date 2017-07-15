import tensorflow as tf
import numpy as np

def leaky_relu(x, alpha=0.01):
    return tf.maximum(alpha * x, x)

def create_cell(dim, n_layers, dropout):
    cell = tf.nn.rnn_cell.GRUCell(dim)
    cell = tf.nn.rnn_cell.DropoutWrapper(cell,
        input_keep_prob=dropout)
    if n_layers > 1:
        cell = tf.nn.rnn_cell.MultiRNNCell([cell] * n_layers)
    return cell

def retrive_var(scopes):
    var = []
    for scope in scopes:
        var += tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
            scope=scope)
    return var

def feed_forward(inp, scope):
    dim = inp.get_shape().as_list()[-1]

    with tf.variable_scope(scope):
        W1 = tf.get_variable('W1', [dim, dim])
        b1 = tf.get_variable('b1', [dim])
        W2 = tf.get_variable('W2', [dim, 1])
        b2 = tf.get_variable('b2', [1])
    h1 = leaky_relu(tf.matmul(inp, W1) + b1)
    logits = tf.matmul(h1, W2) + b2

    return tf.reshape(logits, [-1])

def softmax_word(dropout, proj_W, proj_b, embedding, gamma):

    def loop_func(output):
        output = tf.nn.dropout(output, dropout)
        logits = tf.matmul(output, proj_W) + proj_b
        prob = tf.nn.softmax(logits / gamma)
        inp = tf.matmul(prob, embedding)
        return inp, logits

    return loop_func

def argmax_word(dropout, proj_W, proj_b, embedding):

    def loop_func(output):
        output = tf.nn.dropout(output, dropout)
        logits = tf.matmul(output, proj_W) + proj_b
        word = tf.argmax(logits, axis=1)
        inp = tf.nn.embedding_lookup(embedding, word)
        return inp, logits

    return loop_func

def rnn_decode(h, inp, length, cell, loop_func, scope):
    h_seq, logits_seq = [], []

    with tf.variable_scope(scope):
        tf.get_variable_scope().reuse_variables()
        for t in range(length):
            h_seq.append(tf.expand_dims(h, 1))
            output, h = cell(inp, h)
            inp, logits = loop_func(output)
            logits_seq.append(tf.expand_dims(logits, 1))

    return tf.concat(1, h_seq), tf.concat(1, logits_seq)

def cnn(inp, dropout, scope, reuse=False,
    filter_sizes=[1, 2, 3, 4], num_filters=128):

    dim = inp.get_shape().as_list()[-1]
    inp = tf.expand_dims(inp, -1)

    with tf.variable_scope(scope) as vs:
        if reuse:
            vs.reuse_variables()

        outputs = []
        for size in filter_sizes:
            with tf.variable_scope('conv-maxpool-%s' % size):
                W = tf.get_variable('W', [size, dim, 1, num_filters])
                b = tf.get_variable('b', [num_filters])
                conv = tf.nn.conv2d(inp, W,
                    strides=[1, 1, 1, 1], padding='VALID')
                h = leaky_relu(conv + b)
                pooled = tf.reduce_max(h, reduction_indices=1)   #max pooling over time
                pooled = tf.reshape(pooled, [-1, num_filters])
                outputs.append(pooled)
        outputs = tf.concat(1, outputs)
        outputs = tf.nn.dropout(outputs, dropout)

        with tf.variable_scope('output'):
            W = tf.get_variable('W', [num_filters*len(filter_sizes), 1])
            b = tf.get_variable('b', [1])
            logits = tf.reshape(tf.matmul(outputs, W) + b, [-1])

    return logits

def discriminator(x_real, x_fake, ones, zeros, dropout, scope):
    d_real = cnn(x_real, dropout, scope)
    d_fake = cnn(x_fake, dropout, scope, reuse=True)

    loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        targets=ones, logits=d_real))
    loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        targets=zeros, logits=d_fake))

    return loss_real + loss_fake

def combine(x, y, scope, reuse=False):
    dim_x = x.get_shape().as_list()[-1]
    dim_y = y.get_shape().as_list()[-1]

    with tf.variable_scope(scope) as vs:
        if reuse:
            vs.reuse_variables()

        W = tf.get_variable('W', [dim_x+dim_y, dim_x])
        b = tf.get_variable('b', [dim_x])

    h = tf.matmul(tf.concat(1, [x, y]), W) + b
    return leaky_relu(h)
