import numpy as np

from utils import strip_eos


class Decoder(object):
    def __init__(self, sess, args, vocab, model):
        self.sess, self.vocab, self.model = sess, vocab, model
        self.vocab_array = np.asarray(vocab)

    def rewrite(self, batch):
        model = self.model
        ori, tsf = self.sess.run([model.hard_logits_ori, model.hard_logits_tsf],
            feed_dict={model.dropout: 1, model.batch_size: batch['size'], model.enc_inputs: batch['enc_inputs'],
                       model.dec_inputs: batch['dec_inputs'], model.labels: batch['labels']})

        ori = [[self.closest_node(i) for i in sent] for sent in ori]
        ori = strip_eos(ori)

        tsf = [[self.closest_node(i) for i in sent] for sent in tsf]
        tsf = strip_eos(tsf)

        return ori, tsf

    def closest_node(self, word_embedded):
        dist_2 = np.sum((self.vocab_array - word_embedded)**2, axis=1)
        word_index = np.argmin(dist_2)
        return self.vocab.id2word[word_index]
