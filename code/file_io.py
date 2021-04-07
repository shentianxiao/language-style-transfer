from nltk import word_tokenize, sent_tokenize
import os
import pandas as pd

def load_doc(path):
    data = []
    with open(path) as f:
        for line in f:
            sents = sent_tokenize(line)
            doc = [word_tokenize(sent) for sent in sents]
            data.append(doc)
    return data

def load_sent(path, max_size=-1):
    data = []
    with open(path) as f:
        for line in f:
            if len(data) == max_size:
                break
            data.append(line.split())
    return data

def load_vec(path):
    x = []
    with open(path) as f:
        for line in f:
            p = line.split()
            p = [float(v) for v in p]
            x.append(p)
    return x

def write_doc(docs, sents, path):
    with open(path, 'w') as f:
        index = 0
        for doc in docs:
            for i in range(len(doc)):
                f.write(' '.join(sents[index]))
                f.write('\n' if i == len(doc)-1 else ' ')
                index += 1

def write_sent(sents, path):
    with open(path, 'w') as f:
        for sent in sents:
            f.write(' '.join(sent) + '\n')

def write_vec(vecs, path):
    with open(path, 'w') as f:
        for vec in vecs:
            for i, x in enumerate(vec):
                f.write('%.3f' % x)
                f.write('\n' if i == len(vec)-1 else ' ')


def write_csv(sents_original_0, sents_transfered_0, sents_original_1, sents_transfered_1, path: str, create_dir=False):
    columns = ["original", "transfered", "original_sentiment"]
    output = pd.DataFrame(columns=columns)
    rows = []
    for original, transfer in zip(sents_original_0, sents_transfered_0):
        original = " ".join(original)
        transfer = " ".join(transfer)
        rows.append([original, transfer, "negative"])
    output = output.append(pd.DataFrame(rows, columns=columns))
    rows = []
    for original, transfer in zip(sents_original_1, sents_transfered_1):
        original = " ".join(original)
        transfer = " ".join(transfer)
        rows.append([original, transfer, "positive"])
    output = output.append(pd.DataFrame(rows, columns=columns))
    if create_dir and not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
        print("Created directory {}".format(os.path.dirname(path)))
    if not path.endswith(".csv"):
        path = path + ".csv"
    output.to_csv(path)