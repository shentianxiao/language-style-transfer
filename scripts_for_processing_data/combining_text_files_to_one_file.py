import glob, os

dir = "/mnt/c/Users/kikibox/Documents/Calibre Library/Agatha Christie"


lines = []
os.chdir(dir)
for file in glob.glob("*"):
    os.chdir(dir+"/"+file)
    for file_2 in glob.glob("*.txt"):
        file_path = dir + "/"+ file + "/" + file_2
        with open(file_path) as fp:
            lines_new = fp.readlines()
            lines = lines + lines_new

with open('/mnt/c/Users/kikibox/Documents/NLP/language-style-transfer/code/creating_data_scripts/agatha_christie_one_file.txt', 'w') as f:
    for s in lines:
        f.write(s + '\n')
