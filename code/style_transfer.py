import os
import random
import sys
import time

import numpy as np

import beam_search
import greedy_decoding
from file_io import load_sent, write_sent
from losses import Losses
from nn import *
from options import load_arguments
from utils import *
from vocab import Vocabulary, build_vocab


class Model(object):
    def __init__(self, args, vocab):
        # dim_y = args.dim_y
        dim_z = args.dim_z
        # dim_h = dim_y + dim_z
        dim_emb = args.dim_emb
        n_layers = args.n_layers
        max_len = args.max_seq_length
        filter_sizes = [int(x) for x in args.filter_sizes.split(',')]
        n_filters = args.n_filters
        beta1, beta2 = 0.9, 0.999

        self.dropout = tf.placeholder(tf.float32, name='dropout')
        self.learning_rate = tf.placeholder(tf.float32, name='learning_rate')
        self.rho = tf.placeholder(tf.float32, name='rho')
        self.gamma = tf.placeholder(tf.float32, name='gamma')

        self.batch_len = tf.placeholder(tf.int32, name='batch_len')
        self.batch_size = tf.placeholder(tf.int32, name='batch_size')
        self.enc_inputs = tf.placeholder(tf.int32, [None, None], name='enc_inputs')  # size * len
        self.dec_inputs = tf.placeholder(tf.int32, [None, None], name='dec_inputs')
        self.targets = tf.placeholder(tf.int32, [None, None], name='targets')
        self.weights = tf.placeholder(tf.float32, [None, None], name='weights')
        self.labels = tf.placeholder(tf.float32, [None], name='labels')

        # labels = tf.reshape(self.labels, [-1, 1])

        embedding = tf.get_variable('embedding', initializer=vocab.embedding.astype(np.float32))
        # with tf.variable_scope('projection'):
        #     proj_W = tf.get_variable('W', [dim_h, vocab.size])
        #     proj_b = tf.get_variable('b', [vocab.size])

        enc_inputs = tf.nn.embedding_lookup(embedding, self.enc_inputs)
        dec_inputs = tf.nn.embedding_lookup(embedding, self.dec_inputs)

        targets = tf.nn.embedding_lookup(embedding, self.targets)

        #####   auto-encoder   #####
        init_state = tf.zeros([self.batch_size, dim_z])
        cell_e = create_cell(dim_z, n_layers, self.dropout)
        _, z = tf.nn.dynamic_rnn(cell_e, enc_inputs, initial_state=init_state, scope='encoder')
        z = z

        # cell_e = create_cell(dim_z, n_layers, self.dropout)
        # _, z = tf.nn.dynamic_rnn(cell_e, enc_inputs,
        #    dtype=tf.float32, scope='encoder')

        self.h_ori = z
        self.h_tsf = z

        cell_g = create_cell(dim_z, n_layers, self.dropout)
        g_outputs, _ = tf.nn.dynamic_rnn(cell_g, dec_inputs, initial_state=self.h_ori, scope='generator')

        # attach h0 in the front
        teach_h = tf.concat([tf.expand_dims(self.h_ori, 1), g_outputs], 1)

        g_outputs = tf.nn.dropout(g_outputs, self.dropout)
        # g_outputs = tf.reshape(g_outputs, [-1, dim_z])
        # g_logits = tf.matmul(g_outputs, proj_W) + proj_b
        # labels = tf.reshape(self.targets, [-1])
        reduced_sum = tf.reduce_sum(tf.squared_difference(targets, g_outputs), axis=-1)
        weighted_reduced_sum = reduced_sum * self.weights
        # loss_g = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.reshape(self.targets, [-1]), logits=g_logits)
        # loss_g *= tf.reshape(self.weights, [-1])
        self.loss_g = tf.reduce_sum(weighted_reduced_sum) / tf.to_float(self.batch_size)

        #####   feed-previous decoding   #####
        go = dec_inputs[:, 0, :]
        # soft_func = softsample_word(self.dropout, proj_W, proj_b, embedding, self.gamma)
        # hard_func = argmax_word(self.dropout, proj_W, proj_b, embedding)

        soft_h_ori, soft_logits_ori = rnn_decode(self.h_ori, go, max_len, cell_g, scope='generator')
        soft_h_tsf, soft_logits_tsf = rnn_decode(self.h_tsf, go, max_len, cell_g, scope='generator')

        hard_h_ori, self.hard_logits_ori = rnn_decode(self.h_ori, go, max_len, cell_g, scope='generator')
        hard_h_tsf, self.hard_logits_tsf = rnn_decode(self.h_tsf, go, max_len, cell_g, scope='generator')

        #####   discriminator   #####
        # a batch's first half consists of sentences of one style,
        # and second half of the other
        half = self.batch_size / 2
        zeros, ones = self.labels[:half], self.labels[half:]
        soft_h_tsf = soft_h_tsf[:, :1 + self.batch_len, :]

        self.loss_d0 = discriminator(teach_h[:half], soft_h_tsf[half:], ones, zeros, filter_sizes, n_filters,
                                     self.dropout, scope='discriminator0')
        # self.loss_d1 = discriminator(teach_h[half:], soft_h_tsf[:half], ones, zeros, filter_sizes, n_filters,
        #                              self.dropout, scope='discriminator1')

        #####   optimizer   #####
        self.loss_d = self.loss_d0  # + self.loss_d1
        self.loss = self.loss_g - self.rho * self.loss_d

        theta_eg = retrive_var(['encoder', 'generator', 'embedding', 'projection'])
        theta_d0 = retrive_var(['discriminator0'])
        # theta_d1 = retrive_var(['discriminator1'])

        self.optimizer_all = tf.train.AdamOptimizer(self.learning_rate, beta1, beta2).minimize(self.loss,
                                                                                               var_list=theta_eg)
        self.optimizer_ae = tf.train.AdamOptimizer(self.learning_rate, beta1, beta2).minimize(self.loss_g,
                                                                                              var_list=theta_eg)
        self.optimizer_d0 = tf.train.AdamOptimizer(self.learning_rate, beta1, beta2).minimize(self.loss_d0,
                                                                                              var_list=theta_d0)
        # self.optimizer_d1 = tf.train.AdamOptimizer(self.learning_rate, beta1, beta2).minimize(self.loss_d1,
        #                                                                                       var_list=theta_d1)

        self.saver = tf.train.Saver()


def transfer(model, decoder, sess, args, vocab, data0, data1, out_path):
    batches, order0, order1 = get_batches(data0, data1, vocab.word2id, args.batch_size)

    data0_tsf, data1_tsf = [], []
    losses = Losses(len(batches))
    for batch in batches:
        ori, tsf = decoder.rewrite(batch)
        half = batch['size'] / 2
        data0_tsf += tsf[:half]
        data1_tsf += tsf[half:]

        loss, loss_g, loss_d, loss_d0 = sess.run(
            [model.loss, model.loss_g, model.loss_d, model.loss_d0],
            feed_dict=feed_dictionary(model, batch, args.rho, args.gamma_min))
        losses.add(loss, loss_g, loss_d, loss_d0)

    n0, n1 = len(data0), len(data1)
    data0_tsf = reorder(order0, data0_tsf)[:n0]
    data1_tsf = reorder(order1, data1_tsf)[:n1]

    if out_path:
        write_sent(data0_tsf, out_path + '.0' + '.tsf')
        write_sent(data1_tsf, out_path + '.1' + '.tsf')

    return losses


def create_model(sess, args, vocab):
    model = Model(args, vocab)
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

    vocab = Vocabulary(args.vocab, args.embedding, args.dim_emb)
    print 'vocabulary size:', vocab.size

    if args.dev:
        dev0 = load_sent(args.dev + '.0')
        dev1 = load_sent(args.dev + '.1')

    if args.test:
        test0 = load_sent(args.test + '.0')
        test1 = load_sent(args.test + '.1')

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        model = create_model(sess, args, vocab)

        # TODO: Understand the benefit of beam search
        if args.beam > 1:
            decoder = beam_search.Decoder(sess, args, vocab, model)
        else:
            decoder = greedy_decoding.Decoder(sess, args, vocab, model)

        if args.train:
            batches, _, _ = get_batches(train0, train1, vocab.word2id, args.batch_size)
            random.shuffle(batches)

            start_time = time.time()
            step = 0
            losses = Losses(args.steps_per_checkpoint)
            best_dev = float('inf')
            learning_rate = args.learning_rate
            rho = args.rho
            gamma = args.gamma_init
            dropout = args.dropout_keep_prob

            for epoch in range(1, 1 + args.max_epochs):
                print '--------------------epoch %d--------------------' % epoch
                print 'learning_rate:', learning_rate, '  gamma:', gamma

                for batch in batches:
                    feed_dict = feed_dictionary(model, batch, rho, gamma, dropout, learning_rate)

                    loss_d0, _ = sess.run([model.loss_d0, model.optimizer_d0], feed_dict=feed_dict)
                    # loss_d1, _ = sess.run([model.loss_d1, model.optimizer_d1], feed_dict=feed_dict)

                    # do not back-propagate from the discriminator
                    # when it is too poor
                    if loss_d0 < 1.2:# and loss_d1 < 1.2:
                        optimizer = model.optimizer_all
                    else:
                        optimizer = model.optimizer_ae

                    loss, loss_g, loss_d, _ = sess.run([model.loss, model.loss_g, model.loss_d, optimizer],
                                                       feed_dict=feed_dict)

                    step += 1
                    losses.add(loss, loss_g, loss_d, loss_d0)

                    if step % args.steps_per_checkpoint == 0:
                        losses.output('step %d, time %.0fs,' % (step, time.time() - start_time))
                        losses.clear()

                if args.dev:
                    dev_losses = transfer(model, decoder, sess, args, vocab, dev0, dev1,
                                          args.output + '.epoch%d' % epoch)
                    dev_losses.output('dev')
                    if dev_losses.loss < best_dev:
                        best_dev = dev_losses.loss
                        print 'saving model...'
                        model.saver.save(sess, args.model)

                gamma = max(args.gamma_min, gamma * args.gamma_decay)

        if args.test:
            test_losses = transfer(model, decoder, sess, args, vocab, test0, test1, args.output)
            test_losses.output('test')

        if args.online_testing:
            while True:
                sys.stdout.write('> ')
                sys.stdout.flush()
                inp = sys.stdin.readline().rstrip()
                if inp == 'quit' or inp == 'exit':
                    break
                inp = inp.split()
                y = int(inp[0])
                sent = inp[1:]

                batch = get_batch([sent], [y], vocab.word2id)
                ori, tsf = decoder.rewrite(batch)
                print 'original:', ' '.join(w for w in ori[0])
                print 'transfer:', ' '.join(w for w in tsf[0])
