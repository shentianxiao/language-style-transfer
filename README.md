# Language Style Transfer
This repo contains the code and data of the following paper:

<i> "Style Transfer from Non-Parallel Text by Cross-Alignment". Tianxiao Shen, Tao Lei, Regina Barzilay, and Tommi Jaakkola. [arXiv:1705.09655](https://arxiv.org/abs/1705.09655)<\i>

The method learns to perform style transfer between two non-parallel corpora. For example, given positive and negative reviews as two corpora, the model can learn to modify a sentence's sentiment.

## Quick start
The <code>data/yelp</code> directory contains an example yelp-review dataset. Please name the corpora of two styles by "x.0" and "x.1" respectively. Each file should consist of one sentence per line with tokens separated by a space.

To train a model, first create a <code>tmp/</code> folder, then go to the <code>code/</code> folder and run the following command:
```bash
python style_transfer.py --train ../data/yelp/sentiment.train --dev ../data/yelp/sentiment.dev --output ../tmp/sentiment.dev --vocab ../tmp/yelp.vocab --model ../tmp/model
```

To test the model, run the following command:
```bash
python style_transfer.py --test ../data/yelp/sentiment.test --output ../tmp/sentiment.test --vocab ../tmp/yelp.vocab --model ../tmp/model --load_model true
```

The model and results will be stored in the <code>tmp/</code> folder.

Check <code>code/options.py</code> to see all running options.

## Dependencies
Python >= 2.7, TensorFlow 0.12.0
