import os
import sys
import time
import ipdb
import random
import numpy as np
import tensorflow as tf

from vocab import Vocabulary, build_vocab
from options import load_arguments
from file_io import load_sent
from utils import *
from nn import *

class Model(object):

    def __init__(self, args, vocab):
        dim_emb = args.dim_emb
        dim_z = args.dim_z
        n_layers = args.n_layers

        self.dropout = tf.placeholder(tf.float32,
            name='dropout')
        self.learning_rate = tf.placeholder(tf.float32,
            name='learning_rate')
        self.batch_size = tf.placeholder(tf.int32,
            name='batch_size')
        self.inputs = tf.placeholder(tf.int32, [None, None],    #batch_size * max_len
            name='inputs')
        self.targets = tf.placeholder(tf.int32, [None, None],
            name='targets')
        self.weights = tf.placeholder(tf.float32, [None, None],
            name='weights')

        embedding = tf.get_variable('embedding',
            initializer=vocab.embedding.astype(np.float32))
        with tf.variable_scope('projection'):
            proj_W = tf.get_variable('W', [dim_z, vocab.size])
            proj_b = tf.get_variable('b', [vocab.size])

        inputs = tf.nn.embedding_lookup(embedding, self.inputs)
        cell = create_cell(dim_z, n_layers, self.dropout)
        outputs, _ = tf.nn.dynamic_rnn(cell, inputs,
            dtype=tf.float32, scope='language_model')
        outputs = tf.nn.dropout(outputs, self.dropout)
        outputs = tf.reshape(outputs, [-1, dim_z])
        self.logits = tf.matmul(outputs, proj_W) + proj_b

        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=tf.reshape(self.targets, [-1]),
            logits=self.logits)
        loss *= tf.reshape(self.weights, [-1])
        self.tot_loss = tf.reduce_sum(loss)
        self.sent_loss = self.tot_loss / tf.to_float(self.batch_size)

        self.optimizer = tf.train.AdamOptimizer(self.learning_rate) \
            .minimize(self.sent_loss)

        self.saver = tf.train.Saver()

def create_model(sess, args, vocab):
    model = Model(args, vocab)
    if args.load_model:
        print('Loading model from', args.model)
        model.saver.restore(sess, args.model)
    else:
        print('Creating model with fresh parameters.')
        sess.run(tf.global_variables_initializer())
    return model

def get_lm_batches(x, word2id, batch_size):
    pad = word2id['<pad>']
    go = word2id['<go>']
    eos = word2id['<eos>']
    unk = word2id['<unk>']

    x = sorted(x, key=lambda i: len(i))

    batches = []
    s = 0
    while s < len(x):
        t = min(s + batch_size, len(x))

        go_x, x_eos, weights = [], [], []
        max_len = max([len(sent) for sent in x[s:t]])
        for sent in x[s:t]:
            sent_id = [word2id[w] if w in word2id else unk for w in sent]
            l = len(sent)
            padding = [pad] * (max_len - l)
            go_x.append([go] + sent_id + padding)
            x_eos.append(sent_id + [eos] + padding)
            weights.append([1.0] * l + [0.0] * (max_len-l+1))

        batches.append({'inputs': go_x,
                        'targets': x_eos,
                        'weights': weights,
                        'size': t-s})
        s = t

    return batches

def evaluate(sess, args, vocab, model, x):
    batches = get_lm_batches(x, vocab.word2id, args.batch_size)
    tot_loss, n_words = 0, 0

    for batch in batches:
        tot_loss += sess.run(model.tot_loss,
            feed_dict={model.batch_size: batch['size'],
                       model.inputs: batch['inputs'],
                       model.targets: batch['targets'],
                       model.weights: batch['weights'],
                       model.dropout: 1})
        n_words += np.sum(batch['weights'])

    return np.exp(tot_loss / n_words)

if __name__ == '__main__':
    args = load_arguments()

    if args.train:
        train = load_sent(args.train)

        if not os.path.isfile(args.vocab):
            build_vocab(train, args.vocab)

    vocab = Vocabulary(args.vocab, args.embedding, args.dim_emb)
    print('vocabulary size', vocab.size)

    if args.dev:
        dev = load_sent(args.dev)

    if args.test:
        test = load_sent(args.test)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        model = create_model(sess, args, vocab)
        if args.train:
            batches = get_lm_batches(train, vocab.word2id, args.batch_size)
            random.shuffle(batches)

            start_time = time.time()
            step = 0
            loss = 0.0
            best_dev = float('inf')
            learning_rate = args.learning_rate

            for epoch in range(args.max_epochs):
                print('----------------------------------------------------')
                print('epoch %d, learning_rate %f' % (epoch + 1, learning_rate))

                for batch in batches:
                    step_loss, _ = sess.run([model.sent_loss, model.optimizer],
                        feed_dict={model.batch_size: batch['size'],
                                   model.inputs: batch['inputs'],
                                   model.targets: batch['targets'],
                                   model.weights: batch['weights'],
                                   model.dropout: args.dropout_keep_prob,
                                   model.learning_rate: learning_rate})

                    step += 1
                    loss += step_loss / args.steps_per_checkpoint

                    if step % args.steps_per_checkpoint == 0:
                        print('step %d, time %.0fs, loss %.2f' \
                            % (step, time.time() - start_time, loss))
                        loss = 0.0

                if args.dev:
                    ppl = evaluate(sess, args, vocab, model, dev)
                    print('dev perplexity %.2f' % ppl)
                    if ppl < best_dev:
                        best_dev = ppl
                        unchanged = 0
                        print('Saving model...')
                        model.saver.save(sess, args.model)

        if args.test:
            ppl = evaluate(sess, args, vocab, model, test)
            print('test perplexity %.2f' % ppl)
