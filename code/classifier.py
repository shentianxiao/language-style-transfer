import os
import sys
import time
import ipdb
import random
import numpy as np
import tensorflow as tf

from options import load_arguments
from vocab import Vocabulary, build_vocab
from file_io import load_sent
from nn import cnn

class Model(object):

    def __init__(self, args, vocab):
        dim_emb = args.dim_emb
        filter_sizes = [int(x) for x in args.filter_sizes.split(',')]
        n_filters = args.n_filters

        self.dropout = tf.placeholder(tf.float32,
            name='dropout')
        self.learning_rate = tf.placeholder(tf.float32,
            name='learning_rate')
        self.x = tf.placeholder(tf.int32, [None, None],    #batch_size * max_len
            name='x')
        self.y = tf.placeholder(tf.float32, [None],
            name='y')

        embedding = tf.get_variable('embedding', [vocab.size, dim_emb])
        x = tf.nn.embedding_lookup(embedding, self.x)
        self.logits = cnn(x, filter_sizes, n_filters, self.dropout, 'cnn')
        self.probs = tf.sigmoid(self.logits)

        loss = tf.nn.sigmoid_cross_entropy_with_logits(
            labels=self.y, logits=self.logits)
        self.loss = tf.reduce_mean(loss)
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate) \
            .minimize(self.loss)

        self.saver = tf.train.Saver()

def create_model(sess, args, vocab):
    model = Model(args, vocab)
    if args.load_model:
        print 'Loading model from', args.model
        model.saver.restore(sess, args.model)
    else:
        print 'Creating model with fresh parameters.'
        sess.run(tf.global_variables_initializer())
    return model

def evaluate(sess, args, vocab, model, x, y):
    probs = []
    batches = get_batches(x, y, vocab.word2id, args.batch_size)
    for batch in batches:
        p = sess.run(model.probs,
            feed_dict={model.x: batch['x'],
                       model.dropout: 1})
        probs += p.tolist()
    y_hat = [p > 0.5 for p in probs]
    same = [p == q for p, q in zip(y, y_hat)]
    return 100.0 * sum(same) / len(y), probs

def get_batches(x, y, word2id, batch_size, min_len=5):
    pad = word2id['<pad>']
    unk = word2id['<unk>']

    batches = []
    s = 0
    while s < len(x):
        t = min(s + batch_size, len(x))

        _x = []
        max_len = max([len(sent) for sent in x[s:t]])
        max_len = max(max_len, min_len)
        for sent in x[s:t]:
            sent_id = [word2id[w] if w in word2id else unk for w in sent]
            padding = [pad] * (max_len - len(sent))
            _x.append(padding + sent_id)

        batches.append({'x': _x,
                        'y': y[s:t]})
        s = t

    return batches

def prepare(path, suffix=''):
    data0 = load_sent(path + '.0' + suffix)
    data1 = load_sent(path + '.1' + suffix)
    x = data0 + data1
    y = [0] * len(data0) + [1] * len(data1)
    z = sorted(zip(x, y), key=lambda i: len(i[0]))
    return zip(*z)

if __name__ == '__main__':
    args = load_arguments()

    if args.train:
        train_x, train_y = prepare(args.train)

        if not os.path.isfile(args.vocab):
            build_vocab(train_x, args.vocab)

    vocab = Vocabulary(args.vocab)
    print 'vocabulary size', vocab.size

    if args.dev:
        dev_x, dev_y = prepare(args.dev)

    if args.test:
        test_x, test_y = prepare(args.test)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        model = create_model(sess, args, vocab)
        if args.train:
            batches = get_batches(train_x, train_y,
                vocab.word2id, args.batch_size)
            random.shuffle(batches)

            start_time = time.time()
            step = 0
            loss = 0.0
            best_dev = float('-inf')
            learning_rate = args.learning_rate

            for epoch in range(1, 1+args.max_epochs):
                print '--------------------epoch %d--------------------' % epoch

                for batch in batches:
                    step_loss, _ = sess.run([model.loss, model.optimizer],
                        feed_dict={model.x: batch['x'],
                                   model.y: batch['y'],
                                   model.dropout: args.dropout_keep_prob,
                                   model.learning_rate: learning_rate})

                    step += 1
                    loss += step_loss / args.steps_per_checkpoint

                    if step % args.steps_per_checkpoint == 0:
                        print 'step %d, time %.0fs, loss %.2f' \
                            % (step, time.time() - start_time, loss)
                        loss = 0.0

                if args.dev:
                    acc, _ = evaluate(sess, args, vocab, model, dev_x, dev_y)
                    print 'dev accuracy %.2f' % acc
                    if acc > best_dev:
                        best_dev = acc
                        print 'Saving model...'
                        model.saver.save(sess, args.model)

        if args.test:
            acc, _ = evaluate(sess, args, vocab, model, test_x, test_y)
            print 'test accuracy %.2f' % acc
