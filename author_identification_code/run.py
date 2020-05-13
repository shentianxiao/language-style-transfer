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
hemingway_path_training = "/mnt/c/Users/kikibox/Documents/NLP/language-style-transfer/code/creating_data_scripts/full_hemingway-v2.txt"
shakespeare_path_training = "/mnt/c/Users/kikibox/Documents/NLP/language-style-transfer/code/creating_data_scripts/full_shakespeare-v2.txt"

with open(hemingway_path_training) as f:
    hemingway_lines_training = list(f)

with open(shakespeare_path_training) as f:
    shakespeare_lines_training = list(f)

hemingway_lines_training = [line.replace("\n", "")for line in hemingway_lines_training]
shakespeare_lines_training = [line.replace("\n", "")for line in shakespeare_lines_training]

df_training = pd.DataFrame({'text': shakespeare_lines_training+ hemingway_lines_training, 'label': [0]*len(shakespeare_lines_training) + [1]*len(hemingway_lines_training)})

count_vec_training = CountVectorizer(ngram_range = (1,2), analyzer = "word")
term_doc_mat_training = count_vec_training.fit_transform(df_training['text'])
training_vocab = count_vec_training.vocabulary_

clf = svm.LinearSVC().fit(term_doc_mat_training, df_training['label'])

results = []

for i in range(1,151):
# #"C:\Users\kikibox\Documents\NLP\language-style-transfer\results\shakespeare_150\tmp_shakespeare_hemingway_150_epochs\sentiment.test.epoch150.0.tsf"
    hemingway_path_results = "/mnt/c/Users/kikibox/Documents/NLP/language-style-transfer/results/shakespeare_150/tmp_shakespeare_hemingway_150_epochs/sentiment.test.epoch"+str(i)+".0.tsf"
    shakespeare_path_results = "/mnt/c/Users/kikibox/Documents/NLP/language-style-transfer/results/shakespeare_150/tmp_shakespeare_hemingway_150_epochs/sentiment.test.epoch"+str(i)+".1.tsf"

    with open(hemingway_path_results) as f:
        hemingway_lines_test = list(f)

    with open(shakespeare_path_results) as f:
        shakespeare_lines_test = list(f)


    hemingway_lines_test = [line.replace("\n", "")for line in hemingway_lines_test]
    shakespeare_lines_test = [line.replace("\n", "")for line in shakespeare_lines_test]

    df_testing = pd.DataFrame({'text': shakespeare_lines_test+ hemingway_lines_test, 'label': [0]*len(shakespeare_lines_test) + [1]*len(hemingway_lines_test)})



    count_vec_testing = CountVectorizer(ngram_range = (1,2), analyzer = "word", vocabulary = training_vocab)
    term_doc_mat_testing = count_vec_testing.transform(df_testing['text'])


    labels = list(df_testing['label'])
    random.shuffle(labels)
    df_testing['random_label'] = labels

    df_testing['inversed_label'] = 0
    df_testing.loc[df_testing['label'] == 1, 'inversed_label'] = 0
    df_testing.loc[df_testing['label'] == 0, 'inversed_label'] = 1

    print("RESULTS FOR EPOCH "+str(i))

    predicted= clf.predict(term_doc_mat_testing)

    dict_res = {}
    dict_res['epoch'] = i
    dict_res['prediction_type'] = "predictions on style transfer text"
    dict_res['f1_score_weighted'] = metrics.f1_score(df_testing['label'], predicted ,average = "weighted")
    dict_res['accurary_score'] = metrics.accuracy_score(df_testing['label'], predicted)
    dict_res['precision_score_weighted'] = metrics.precision_score(df_testing['label'], predicted, average = "weighted")
    dict_res['recall_score_weighted'] = metrics.f1_score(df_testing['label'], predicted ,average = "weighted")
    results.append(dict_res)

    print("MultinomialNB Accuracy predictions:",metrics.accuracy_score(df_testing['label'], predicted))
    print("MultinomialNB Precision predictions:",metrics.precision_score(df_testing['label'], predicted, average = "weighted"))
    print("MultinomialNB Recall predictions:",metrics.recall_score(df_testing['label'], predicted ,average = "weighted"))
    print("MultinomialNB F1-Score predictions:",metrics.f1_score(df_testing['label'], predicted ,average = "weighted"))

    dict_res = {}
    dict_res['epoch'] = i
    dict_res['prediction_type'] = "random baseline"
    dict_res['f1_score_weighted'] = metrics.f1_score(df_testing['label'], df_testing['random_label'], average = "weighted")
    dict_res['accurary_score'] = metrics.accuracy_score(df_testing['label'], df_testing['random_label'])
    dict_res['precision_score_weighted'] = metrics.precision_score(df_testing['label'], df_testing['random_label'], average = "weighted")
    dict_res['recall_score_weighted'] = metrics.f1_score(df_testing['label'], df_testing['random_label'], average = "weighted")
    results.append(dict_res)

    print("MultinomialNB Accuracy random baseline:",metrics.accuracy_score(df_testing['label'], df_testing['random_label']))
    print("MultinomialNB Precision random baseline:",metrics.precision_score(df_testing['label'], df_testing['random_label'], average = "weighted"))
    print("MultinomialNB Recall random baseline:",metrics.recall_score(df_testing['label'], df_testing['random_label'], average = "weighted"))
    print("MultinomialNB F1-score random baseline:",metrics.f1_score(df_testing['label'], df_testing['random_label'], average = "weighted"))

    dict_res = {}
    dict_res['epoch'] = i
    dict_res['prediction_type'] = "inverted baseline"
    dict_res['f1_score_weighted'] = metrics.f1_score(df_testing['inversed_label'], predicted, average = "weighted")
    dict_res['accurary_score'] = metrics.accuracy_score(df_testing['inversed_label'], predicted)
    dict_res['precision_score_weighted'] = metrics.precision_score(df_testing['inversed_label'], predicted, average = "weighted")
    dict_res['recall_score_weighted'] = metrics.recall_score(df_testing['inversed_label'], predicted, average = "weighted")
    results.append(dict_res)

    print("MultinomialNB Accuracy inverted labels:",metrics.accuracy_score(df_testing['inversed_label'], predicted))
    print("MultinomialNB Precision inverted:",metrics.precision_score(df_testing['inversed_label'], predicted, average = "weighted"))
    print("MultinomialNB Recall inverted:",metrics.recall_score(df_testing['inversed_label'], predicted, average = "weighted"))
    print("MultinomialNB F1-score inverted:",metrics.f1_score(df_testing['inversed_label'], predicted, average = "weighted"))

df_results = pd.DataFrame.from_records(results)
df_results.to_csv("shakespeare_hemingway_f1_score_overtime.csv")
