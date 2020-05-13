import re
import random
import unicodedata

#C:\Users\kikibox\Documents\NLP\language-style-transfer\data_authors
file_path = "/mnt/c/Users/kikibox/Documents/NLP/language-style-transfer/data_authors/charlesdickens"

with open(file_path) as f:
    lines = f.readlines()

full_lines = " ".join(lines)
full_lines = full_lines.lower()
full_lines = full_lines.replace("\n","")
full_lines = full_lines.replace("\r","")
#idx = 0
#for line in lines:
#    line = line.lower()
#    line = line.replace("\n","")
#    line = line.replace("\r","")
#    full_lines = full_lines + " " + line
#    if idx % 10000 == 0:
#        print('first pass : ' + str(idx) + " out of "+ str(len(lines)))
#    idx = idx + 1
full_lines = full_lines.decode('utf8').encode('ascii', errors='ignore')
sentences = re.findall('.*?[.!\?]', full_lines)

final_sentences = []
idx = 0
len_sentences = len(sentences)
for sentence in sentences:
    sentence = sentence.replace(',', " , ")
    sentence = sentence.replace('.', " . ")
    sentence = sentence.replace('?', " ? ")
    sentence = sentence.replace('!', " ! ")
    sentence = sentence.replace('$', " $ ")
    sentence = sentence.replace(';', " ; ")
    sentence = sentence.replace('\'', " \' ")
    sentence = sentence.replace('\"', " \" ")
    sentence = sentence.replace('\\', " \\ ")
    sentence = sentence.replace(':', " : ")
    sentence = re.sub(r'\d+'," _num_ ", sentence)
    words = sentence.split(" ")
    final_words = []
    for w in words:
        if w.strip() != "" and w:
            final_words.append(w)
    str_s = " ".join(final_words)
    final_sentences.append(str_s)

    if idx % 10000 == 0:
        print('second pass : ' + str(idx) + " out of "+ str(len_sentences))
    idx = idx + 1

#print(final_sentences)
unique_list_final_sentences = list(set(final_sentences))
len_copora = len(unique_list_final_sentences)
random.shuffle(unique_list_final_sentences)
training_up_off = int(len_copora * .9)
training_set = unique_list_final_sentences[0:training_up_off]
test_set = unique_list_final_sentences[(training_up_off): len(unique_list_final_sentences)]

with open('training_charles_dickens-v2_full_sentence.txt', 'w') as filehandle:
    for listitem in training_set:
        filehandle.write('%s\n' % listitem)


with open('testing_charles_dickens-v2_full_sentence.txt', 'w') as filehandle:
    for listitem in test_set:
        filehandle.write('%s\n' % listitem)
