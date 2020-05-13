import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import RegexpTokenizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
import random
from sklearn import svm

# followed the following tutorial
# https://www.datacamp.com/community/tutorials/text-analytics-beginners-nltk

# reading in hemingway

# "C:\Users\kikibox\Documents\NLP\language-style-transfer\code\creating_data_scripts\non_full_sentences_all_data_agatha_christie.txt"
hemingway_path_training = "/mnt/c/Users/kikibox/Documents/NLP/language-style-transfer/code/creating_data_scripts/non_full_sentences_all_data_agatha_christie.txt"
# "C:\Users\kikibox\Documents\NLP\language-style-transfer\code\creating_data_scripts\non_full_sentences_charles_dickens_full_data.txt"
shakespeare_path_training = "/mnt/c/Users/kikibox/Documents/NLP/language-style-transfer/code/creating_data_scripts/non_full_sentences_charles_dickens_full_data.txt"

hemingway_path_results = "/mnt/c/Users/kikibox/Documents/NLP/language-style-transfer/results/christie_dickens_not_full_sentence_8_epochs/tmp/sentiment.test.epoch8.0.tsf"
# "C:\Users\kikibox\Documents\NLP\language-style-transfer\results\christie_dickens_not_full_sentence_8_epochs\tmp\sentiment.test.epoch8.0.tsf"
shakespeare_path_results = "/mnt/c/Users/kikibox/Documents/NLP/language-style-transfer/results/christie_dickens_not_full_sentence_8_epochs/tmp/sentiment.test.epoch8.1.tsf"

with open(hemingway_path_training) as f:
    hemingway_lines_training = list(f)

with open(shakespeare_path_training) as f:
    shakespeare_lines_training = list(f)

with open(hemingway_path_results) as f:
    hemingway_lines_test = list(f)

with open(shakespeare_path_results) as f:
    shakespeare_lines_test = list(f)

hemingway_lines_training = [line.replace("\n", "")for line in hemingway_lines_training]
shakespeare_lines_training = [line.replace("\n", "")for line in shakespeare_lines_training]
hemingway_lines_test = [line.replace("\n", "")for line in hemingway_lines_test]
shakespeare_lines_test = [line.replace("\n", "")for line in shakespeare_lines_test]


df_training = pd.DataFrame({'text': shakespeare_lines_training+ hemingway_lines_training, 'label': [0]*len(shakespeare_lines_training) + [1]*len(hemingway_lines_training)})
df_testing = pd.DataFrame({'text': shakespeare_lines_test+ hemingway_lines_test, 'label': [0]*len(shakespeare_lines_test) + [1]*len(hemingway_lines_test)})
# making count CountVectorizer



count_vec_training = CountVectorizer(ngram_range = (1,2), analyzer = "word")
term_doc_mat_training = count_vec_training.fit_transform(df_training['text'])
training_vocab = count_vec_training.vocabulary_


count_vec_testing = CountVectorizer(ngram_range = (1,2), analyzer = "word", vocabulary = training_vocab)
term_doc_mat_testing = count_vec_testing.transform(df_testing['text'])


#X_train, X_test, y_train, y_test = train_test_split(term_doc_mat, df['label'], test_size=0.1)

#X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=0.112)

#print(X_train.shape)
#print(X_validation.shape)
#print(X_test.shape)

labels = list(df_testing['label'])
random.shuffle(labels)
df_testing['random_label'] = labels

df_testing['inversed_label'] = 0
df_testing.loc[df_testing['label'] == 1, 'inversed_label'] = 0
df_testing.loc[df_testing['label'] == 0, 'inversed_label'] = 1

clf = svm.LinearSVC().fit(term_doc_mat_training, df_training['label'])
predicted= clf.predict(term_doc_mat_testing)
print("MultinomialNB Accuracy:",metrics.accuracy_score(df_testing['label'], predicted))
print("MultinomialNB Precision:",metrics.precision_score(df_testing['label'], predicted, average = "weighted"))
print("MultinomialNB Recall:",metrics.recall_score(df_testing['label'], predicted ,average = "weighted"))
print("MultinomialNB F1-Score:",metrics.f1_score(df_testing['label'], predicted ,average = "weighted"))



print("MultinomialNB Accuracy:",metrics.accuracy_score(df_testing['label'], df_testing['random_label']))
print("MultinomialNB Precision:",metrics.precision_score(df_testing['label'], df_testing['random_label'], average = "weighted"))
print("MultinomialNB Recall:",metrics.recall_score(df_testing['label'], df_testing['random_label'], average = "weighted"))
print("MultinomialNB F1-score:",metrics.f1_score(df_testing['label'], df_testing['random_label'], average = "weighted"))

print("MultinomialNB Accuracy:",metrics.accuracy_score(df_testing['inversed_label'], predicted))
print("MultinomialNB Precision:",metrics.precision_score(df_testing['inversed_label'], predicted, average = "weighted"))
print("MultinomialNB Recall:",metrics.recall_score(df_testing['inversed_label'], predicted, average = "weighted"))
print("MultinomialNB F1-score:",metrics.f1_score(df_testing['inversed_label'], predicted, average = "weighted"))
