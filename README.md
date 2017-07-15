# Language Style Transfer

## Quick start

```bash
python style_transfer.py --train ../data/yelp/sentiment.train --dev ../data/yelp/sentiment.dev --output ../tmp/sentiment.dev --vocab ../tmp/yelp.vocab --model ../tmp/model
```

```bash
python style_transfer.py --test ../data/yelp/sentiment.test --output ../tmp/sentiment.test --vocab ../tmp/yelp.vocab --model ../tmp/model --load_model true
```

## Dependencies
Python >= 2.7, TensorFlow 0.12.0

## References
Tianxiao Shen, Tao Lei, Regina Barzilay, and Tommi Jaakkola. Style transfer from non-parallel text by cross-alignment. [arXiv:1705.09655](https://arxiv.org/abs/1705.09655), 2017.
