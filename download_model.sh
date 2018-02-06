mkdir -p model

dir="http://people.csail.mit.edu/tianxiao/language-style-transfer/model/"

wget ${dir}yelp.d100.emb.txt -P model/
wget ${dir}yelp.vocab -P model/
wget ${dir}model.data-00000-of-00001 -P model/
wget ${dir}model.index -P model/
wget ${dir}model.meta -P model/
