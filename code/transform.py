import sys
import time

import beam_search
from accumulator import Accumulator
from vocab import Vocabulary
from nn import *
from options import load_arguments
from style_transfer import create_model, transfer
from utils import *


def transform_text(text):
    tf.compat.v1.disable_eager_execution()
    args = load_arguments()
    ah = vars(args)
    ah['vocab'] = '../model/yelp.vocab'
    ah['model'] = '../model/model'
    ah['load_model'] = True
    ah['beam'] = 8
    ah['batch_size'] = 1
    inp = [text]

    vocab = Vocabulary(args.vocab, args.embedding, args.dim_emb)
    print('vocabulary size:', vocab.size)

    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True

    with tf.compat.v1.Session(config=config) as sess:
        model = create_model(sess, args, vocab)
        decoder = beam_search.Decoder(sess, args, vocab, model)

        '''test_losses = transfer(model, decoder, sess, args, vocab,
                               test0, test1, args.output)'''

        batches, order0, order1 = get_batches(inp, inp,
                                              vocab.word2id, args.batch_size)

        data0_tsf, data1_tsf = [], []
        losses = Accumulator(len(batches), ['loss', 'rec', 'adv', 'd0', 'd1'])

        # rec, tsf = decoder.rewrite(inp)

        # print(rec)
        # print(tsf)
        for batch in batches:
            rec, tsf = decoder.rewrite(batch)
            half = batch['size'] // 2
            print("rec:")
            print(rec)
            print("tsf:")
            print(tsf)
            data0_tsf += tsf[:half]
            data1_tsf += tsf[half:]
        n0, n1 = len(inp), len(inp)
        data0_tsf = reorder(order0, data0_tsf)[:n0]
        data1_tsf = reorder(order1, data1_tsf)[:n1]
        print(data0_tsf)
        print(data1_tsf)

if __name__ == '__main__':
    transform_text(input("type input lol: "))
