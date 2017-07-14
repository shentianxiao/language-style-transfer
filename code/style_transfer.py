import os
import sys
import time
import ipdb
import random
import cPickle as pickle
import numpy as np
import tensorflow as tf

from vocab import Vocabulary, build_vocab
from losses import Losses
from options import load_arguments
from file_io import load_sent, write_sent
from utils import *
from nn import *

class Model(object):

    def __init__(self, sess, args, vocab):
        dim_y = args.dim_y
        dim_z = args.dim_z
        dim_h = dim_y + dim_z
        dim_emb = args.dim_emb
        n_layers = args.n_layers
        max_len = args.max_seq_length
        gamma = args.gamma

        self.dropout = tf.placeholder(tf.float32,
            name='dropout')
        self.learning_rate = tf.placeholder(tf.float32,
            name='learning_rate')
        self.rho = tf.placeholder(tf.float32,
            name='rho')

        self.batch_len = tf.placeholder(tf.int32,
            name='batch_len')
        self.batch_size = tf.placeholder(tf.int32,
            name='batch_size')
        self.enc_inputs = tf.placeholder(tf.int32, [None, None],    #size * len
            name='enc_inputs')
        self.dec_inputs = tf.placeholder(tf.int32, [None, None],
            name='dec_inputs')
        self.targets = tf.placeholder(tf.int32, [None, None],
            name='targets')
        self.weights = tf.placeholder(tf.float32, [None, None],
            name='weights')
        self.labels = tf.placeholder(tf.float32, [None],
            name='labels')

        with tf.variable_scope('encoder_y'):
            y_W = tf.get_variable('W', [1, dim_y])
            y_b = tf.get_variable('b', [dim_y])
        labels = tf.reshape(self.labels, [-1, 1])
        y_ori = tf.matmul(labels, y_W) + y_b
        y_tsf = tf.matmul(1-labels, y_W) + y_b

        embedding = tf.get_variable('embedding', [vocab.size, dim_emb])
        with tf.variable_scope('projection'):
            proj_W = tf.get_variable('W', [dim_h, vocab.size])
            proj_b = tf.get_variable('b', [vocab.size])

        enc_inputs = tf.nn.embedding_lookup(embedding, self.enc_inputs)
        dec_inputs = tf.nn.embedding_lookup(embedding, self.dec_inputs)

        #####   auto-encoder   #####
        cell_e = create_cell(dim_h, n_layers, self.dropout)
        init_state = tf.concat(1, [y_ori, tf.zeros([self.batch_size, dim_z])])
        _, z = tf.nn.dynamic_rnn(cell_e, enc_inputs,
            initial_state=init_state, scope='encoder_z')
        z = z[:, dim_y:]
        #cell_e = create_cell(dim_z, n_layers, self.dropout)
        #_, z = tf.nn.dynamic_rnn(cell_e, enc_inputs,
        #    dtype=tf.float32, scope='encoder_z')

        h_ori = tf.concat(1, [y_ori, z])
        h_tsf = tf.concat(1, [y_tsf, z])
        #h_ori = combine(z, y_ori, scope='generator')
        #h_tsf = combine(z, y_tsf, scope='generator', reuse=True)

        cell_g = create_cell(dim_h, n_layers, self.dropout)
        g_outputs, _ = tf.nn.dynamic_rnn(cell_g, dec_inputs,
            initial_state=h_ori, scope='generator')

        # attach h0 in the front
        teach_h = tf.concat(1, [tf.expand_dims(h_ori, 1), g_outputs])

        g_outputs = tf.nn.dropout(g_outputs, self.dropout)
        g_outputs = tf.reshape(g_outputs, [-1, dim_h])
        g_logits = tf.matmul(g_outputs, proj_W) + proj_b

        loss_g = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=tf.reshape(self.targets, [-1]), logits=g_logits)
        loss_g *= tf.reshape(self.weights, [-1])
        self.loss_g = tf.reduce_sum(loss_g) / tf.to_float(self.batch_size)

        #####   feed-previous decoding   #####
        go = dec_inputs[:,0,:]
        soft_func = softmax_word(self.dropout, proj_W, proj_b, embedding, gamma)
        hard_func = argmax_word(self.dropout, proj_W, proj_b, embedding)

        soft_h_ori, soft_logits_ori = rnn_decode(h_ori, go, max_len,
            cell_g, soft_func, scope='generator')
        soft_h_tsf, soft_logits_tsf = rnn_decode(h_tsf, go, max_len,
            cell_g, soft_func, scope='generator')

        hard_h_ori, self.hard_logits_ori = rnn_decode(h_ori, go, max_len,
            cell_g, hard_func, scope='generator')
        hard_h_tsf, self.hard_logits_tsf = rnn_decode(h_tsf, go, max_len,
            cell_g, hard_func, scope='generator')

        #####   discriminator   #####
        half = self.batch_size / 2
        zeros, ones = self.labels[:half], self.labels[half:]
        soft_h_tsf = soft_h_tsf[:, :1+self.batch_len, :]

        self.loss_d0 = discriminator(teach_h[:half], soft_h_tsf[half:],
            ones, zeros, self.dropout, scope='discriminator0')
        self.loss_d1 = discriminator(teach_h[half:], soft_h_tsf[:half],
            ones, zeros, self.dropout, scope='discriminator1')

        #####   optimizer   #####
        self.loss_d = self.loss_d0 + self.loss_d1
        self.loss = self.loss_g - self.rho * self.loss_d

        theta_eg = retrive_var(['encoder_y', 'encoder_z', 'generator',
            'embedding', 'projection'])
        theta_d0 = retrive_var(['discriminator0'])
        theta_d1 = retrive_var(['discriminator1'])

        self.optimizer_all = tf.train.AdamOptimizer(self.learning_rate) \
            .minimize(self.loss, var_list=theta_eg)
        self.optimizer_ae = tf.train.AdamOptimizer(self.learning_rate) \
            .minimize(self.loss_g, var_list=theta_eg)
        self.optimizer_d0 = tf.train.AdamOptimizer(self.learning_rate) \
            .minimize(self.loss_d0, var_list=theta_d0)
        self.optimizer_d1 = tf.train.AdamOptimizer(self.learning_rate) \
            .minimize(self.loss_d1, var_list=theta_d1)

        self.saver = tf.train.Saver()

def rewrite(model, sess, args, vocab, batch):
    logits_ori, logits_tsf, loss, loss_g, loss_d, loss_d0, loss_d1 = sess.run(
        [model.hard_logits_ori, model.hard_logits_tsf,
         model.loss, model.loss_g, model.loss_d, model.loss_d0, model.loss_d1],
         feed_dict=feed_dictionary(model, batch, args.rho))

    ori = np.argmax(logits_ori, axis=2).tolist()
    ori = [[vocab.id2word[i] for i in sent] for sent in ori]
    ori = strip_eos(ori)

    tsf = np.argmax(logits_tsf, axis=2).tolist()
    tsf = [[vocab.id2word[i] for i in sent] for sent in tsf]
    tsf = strip_eos(tsf)

    return ori, tsf, loss, loss_g, loss_d, loss_d0, loss_d1

def transfer(model, sess, args, vocab, data0, data1, out_path):
    batches, order0, order1 = get_batches(data0, data1,
        vocab.word2id, args.batch_size)

    data0_tsf, data1_tsf = [], []
    losses = Losses(len(batches))
    for batch in batches:
        ori, tsf, loss, loss_g, loss_d, loss_d0, loss_d1 = rewrite(
            model, sess, args, vocab, batch)
        half = batch['size'] / 2
        data0_tsf += tsf[:half]
        data1_tsf += tsf[half:]
        losses.add(loss, loss_g, loss_d, loss_d0, loss_d1)

    n0, n1 = len(data0), len(data1)
    data0_tsf = reorder(order0, data0_tsf)[:n0]
    data1_tsf = reorder(order1, data1_tsf)[:n1]

    if out_path:
        write_sent(data0_tsf, out_path+'.0'+'.tsf')
        write_sent(data1_tsf, out_path+'.1'+'.tsf')

    return losses

def create_model(sess, args, vocab):
    model = Model(sess, args, vocab)
    if args.load_model:
        print 'Loading model from', args.model
        model.saver.restore(sess, args.model)
    else:
        print 'Creating model with fresh parameters.'
        sess.run(tf.global_variables_initializer())
    return model

if __name__ == '__main__':
    args = load_arguments()

    #####   data preparation   #####
    if args.train:
        train0 = load_sent(args.train + '.0', args.max_train_size)
        train1 = load_sent(args.train + '.1', args.max_train_size)
        print '#sents of training file 0:', len(train0)
        print '#sents of training file 1:', len(train1)

        if not os.path.isfile(args.vocab):
            build_vocab(train0 + train1, args.vocab)

    vocab = Vocabulary(args.vocab)
    print 'vocabulary size:', vocab.size

    if args.dev:
        dev0 = load_sent(args.dev + '.0')
        dev1 = load_sent(args.dev + '.1')

    if args.test:
        test0 = load_sent(args.test + '.0')
        test1 = load_sent(args.test + '.1')

    config = tf.ConfigProto()
    #config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        model = create_model(sess, args, vocab)

        if args.train:
            batches, _, _ = get_batches(train0, train1, vocab.word2id,
                args.batch_size)
            random.shuffle(batches)

            start_time = time.time()
            step = 0
            losses = Losses(args.steps_per_checkpoint)
            best_dev = float('inf')
            learning_rate = args.learning_rate

            for epoch in range(1, 1+args.max_epochs):
                print '--------------------epoch %d--------------------' % epoch

                for batch in batches:
                    feed_dict = feed_dictionary(model, batch, args.rho,
                        args.dropout_keep_prob, learning_rate)

                    loss_d0, _ = sess.run([model.loss_d0, model.optimizer_d0],
                        feed_dict=feed_dict)
                    loss_d1, _ = sess.run([model.loss_d1, model.optimizer_d1],
                        feed_dict=feed_dict)

                    # do not back-propagate from the discriminator
                    # when it is too poor
                    if loss_d0 < 0.6 and loss_d1 < 0.6:
                        optimizer = model.optimizer_all
                    else:
                        optimizer = model.optimizer_ae

                    loss, loss_g, loss_d, _ = sess.run(
                        [model.loss, model.loss_g, model.loss_d, optimizer],
                        feed_dict=feed_dict)

                    step += 1
                    losses.add(loss, loss_g, loss_d, loss_d0, loss_d1)

                    if step % args.steps_per_checkpoint == 0:
                        losses.output('step %d, time %.0fs,'
                            % (step, time.time() - start_time))
                        losses.clear()

                if args.dev:
                    dev_losses = transfer(model, sess, args, vocab,
                        dev0, dev1, args.output + '.epoch%d' % epoch)
                    dev_losses.output('dev')
                    if dev_losses.loss < best_dev:
                        best_dev = dev_losses.loss
                        print 'saving model...'
                        model.saver.save(sess, args.model)

        if args.test:
            test_losses = transfer(model, sess, args, vocab,
                test0, test1, args.output)
            test_losses.output('test')
