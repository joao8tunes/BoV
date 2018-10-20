#!/usr/bin/python3.4
# -*- coding: utf-8 -*-
################################################################################
##              Laboratory of Computational Intelligence (LABIC)              ##
##             --------------------------------------------------             ##
##       Originally developed by: João Antunes  (joao8tunes@gmail.com)        ##
##       Laboratory: labic.icmc.usp.br    Personal: joaoantunes.esy.es        ##
##                                                                            ##
##   "Não há nada mais trabalhoso do que viver sem trabalhar". Seu Madruga    ##
################################################################################

import filecmp
import datetime
import argparse
import codecs
import logging
import nltk
import os
import sys
import time
import math
import re


################################################################################
### FUNCTIONS                                                                ###
################################################################################

# Print iterations progress: https://gist.github.com/aubricus/f91fb55dc6ba5557fbab06119420dd6a
def print_progress(iteration, total, estimation, prefix='   ', decimals=1, bar_length=100, final=False):
    columns = 32    #columns = os.popen('stty size', 'r').read().split()[1]    #Doesn't work with nohup.
    eta = str( datetime.timedelta(seconds=max(0, int( math.ceil(estimation) ))) )
    bar_length = int(columns)
    str_format = "{0:." + str(decimals) + "f}"
    percents = str_format.format(100 * (iteration / float(total)))
    filled_length = int(round(bar_length * iteration / float(total)))
    bar = '█' * filled_length + '-' * (bar_length - filled_length)
    sys.stdout.write('\r%s %s%s |%s| %s' % (prefix, percents, '%', bar, eta))

    if final == True:    #iteration == total
        sys.stdout.write('\n')

    sys.stdout.flush()


#Format a value in seconds to "day, HH:mm:ss".
def format_time(seconds):
    return str( datetime.timedelta(seconds=max(0, int( math.ceil(seconds) ))) )


#Convert a string value to boolean:
def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError("invalid boolean value: " + "'" + v + "'")


#Verify if a value correspond to a natural number (it's an integer and bigger than 0):
def natural(v):
    try:
        v = int(v)

        if v > 0:
            return v
        else:
            raise argparse.ArgumentTypeError("invalid natural number value: " + "'" + v + "'")
    except ValueError:
        raise argparse.ArgumentTypeError("invalid natural number value: " + "'" + v + "'")


#Verify if a string correspond to a common word (has just digits, letters (accented or not), hyphens and underlines):
def isword(word):
    if not any( l.isalpha() for l in word ):
        return False

    return all( l.isalpha() or bool(re.search("[A-Za-z0-9-_\']+", l)) for l in word )

################################################################################


################################################################################

#URL: https://github.com/joao8tunes/BoV

#Example usage: python3 BoV.py --language EN --model models/model.txt --input in/db/ --output out/BoV/txt/

#Pre-trained language models:
#English Wikipedia: http://sites.labic.icmc.usp.br/MSc-Thesis_Antunes_2018/input/language-models/W2V-CBoW_Wikipedia/EN/2017-09-26/
#Portuguese Wikipedia: http://sites.labic.icmc.usp.br/MSc-Thesis_Antunes_2018/input/language-models/W2V-CBoW_Wikipedia/PT/2017-09-26/

#Pre-trained word and phrase vectors (Google): https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit?usp=sharing
#More info: https://code.google.com/archive/p/word2vec/

#Defining script arguments:
parser = argparse.ArgumentParser(description="BoV based text representation generator\n=======================================")
parser._action_groups.pop()
required = parser.add_argument_group('required arguments')
optional = parser.add_argument_group('optional arguments')
optional.add_argument("--log", metavar='BOOL', type=str2bool, action="store", dest="log", nargs="?", const=True, default=False, required=False, help='display log during the process: y, [N]')
optional.add_argument("--tokenize", metavar='BOOL', type=str2bool, action="store", dest="tokenize", nargs="?", const=True, default=False, required=False, help='specify if texts need to be tokenized: y, [N]')
optional.add_argument("--ignore_case", metavar='BOOL', type=str2bool, action="store", dest="ignore_case", nargs="?", const=True, default=False, required=False, help='ignore case: y, [N]')
optional.add_argument("--validate_words", metavar='BOOL', type=str2bool, action="store", dest="validate_words", nargs="?", const=True, default=True, required=False, help='validate vocabulary ([A-Za-z0-9-_\']+): [Y], n')
optional.add_argument("--stoplist", metavar='FILE_PATH', type=str, action="store", dest="stoplist", default=None, required=False, nargs="?", const=True, help='specify stoplist file')
optional.add_argument("--n_gram", metavar='INT', type=natural, action="store", dest="n_gram", default=1, nargs="?", const=True, required=False, help='specify max. (>= 1) N-gram: [1]')
required.add_argument("--language", metavar='STR', type=str, action="store", dest="language", nargs="?", const=True, required=True, help='language of database: EN, ES, FR, DE, IT, PT')
required.add_argument("--model", "-m", metavar='FILE_PATH', type=str, action="store", dest="model", required=True, nargs="?", const=True, help='input file of model (Word2Vec text vectors)')
required.add_argument("--input", "-i", metavar='DIR_PATH', type=str, action="store", dest="input", required=True, nargs="?", const=True, help='input directory of database')
required.add_argument("--output", "-o", metavar='DIR_PATH', type=str, action="store", dest="output", required=True, nargs="?", const=True, help='output directory to save the BoVs')
args = parser.parse_args()    #Verifying arguments.

################################################################################


################################################################################

#Setup logging:
if args.log:
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

if args.language == "ES":      #Spanish.
    nltk_language = "spanish"
elif args.language == "FR":    #French.
    nltk_language = "french"
elif args.language == "DE":    #Deutsch.
    nltk_language = "german"
elif args.language == "IT":    #Italian.
    nltk_language = "italian"
elif args.language == "PT":    #Portuguese.
    nltk_language = "portuguese"
else:                          #English.
    args.language = "EN"
    nltk_language = "english"

total_start = time.time()

################################################################################


################################################################################
### INPUT (LOAD DATABASE AND MODEL)                                          ###
################################################################################

log = codecs.open("BoV-log_" + time.strftime("%Y-%m-%d") + "_" + time.strftime("%H-%M-%S") + "_" + str(uuid.uuid4().hex) + ".txt", "w", "utf-8")
print("\nBoV based text representation generator\n=======================================\n\n\n")
log.write("BoV based text representation generator\n=======================================\n\n\n")
log.write("> Parameters:\n")

if args.tokenize:
    log.write("\t- Tokenize:\t\tyes\n")
else:
    log.write("\t- Tokenize:\t\tno\n")

if args.ignore_case:
    log.write("\t- Ignore case:\t\tyes\n")
else:
    log.write("\t- Ignore case:\t\tno\n")

if args.validate_words:
    log.write("\t- Validate words:\tyes\n")
else:
    log.write("\t- Validate words:\tno\n")

if args.stoplist is not None:
    log.write("\t- Stoplist:\t\t" + args.stoplist + "\n")

log.write("\t- N-grams:\t\t<= " + str(args.n_gram) + "\n")
log.write("\t- Language:\t\t" + args.language + "\n")
log.write("\t- Model:\t\t" + args.model + "\n")
log.write("\t- Input:\t\t" + args.input + "\n")
log.write("\t- Output:\t\t" + args.output + "\n\n\n")

if not os.path.exists(args.input):
    print("ERROR: input directory does not exists!\n\t!Directory: " + args.input)
    log.write("ERROR: input directory does not exists!\n\t!Directory: " + args.input)
    log.close()
    sys.exit()

print("> Loading model...\n\n\n")
model = open(args.model, "r")
model_dim = int( model.readline().split()[1] )
vectors = {}

#Loading model as an indexed dictionary:
for vector in model:
    data = vector.strip().split()
    head = data[0].strip()
    data.pop(0)
    vectors[head] = [float(elt) for elt in data]

model.close()
stoplist = []

if args.stoplist is not None:
    print("> Loading stoplist...\n\n\n")
    stoplist_file = codecs.open(args.stoplist, "r", encoding='utf-8')

    for line in stoplist_file.readlines():
        stoplist.append(line.strip())

    if args.ignore_case:
        stoplist = [w.lower() for w in stoplist]

    stoplist.sort()
    stoplist_file.close()

################################################################################

print("> Loading input filepaths...\n\n\n")
files_list = []

#Loading all filepaths from all root directories:
for directory in os.listdir(args.input):
    for file_item in os.listdir(args.input + "/" + directory):
        files_list.append(args.input + directory + "/" + file_item)

files_list.sort()
total_num_examples = len(files_list)
log.write("> Database: " + args.input + "\n")
log.write("\t# Files: " + str(total_num_examples) + "\n\n")

#Reading files:
for filepath in files_list:
    log.write("\t" + filepath + "\n")

################################################################################


################################################################################
### TASK 1 - N-GRAM VARIATION                                                ###
################################################################################

out_string = args.output + "BoV_1st-2nd_ng"
header = str(total_num_examples) + " " + str(model_dim) + "\n"

for dim in range(1, model_dim+1):
    header += "d" + str(dim) + "\t"

header += "class_atr\n"
print("> TASK 1 - N-GRAM VARIATION / TASK 2 - TEXT REPRESENTATION:")
print("..................................................")
total_operations = args.n_gram*total_num_examples
total_num_paragraphs = 0
total_num_sentences = 0
filepath_i = 0
eta = 0
print_progress(filepath_i, total_operations, eta)
operation_start = time.time()

for n in range(1, args.n_gram+1):
    out_file = open(out_string + str(n), "w")
    out_file.write(header)

    for filepath in files_list:
        start = time.time()
        file_item = codecs.open(filepath, "r", encoding='utf-8')
        paragraphs = [p.strip() for p in file_item.readlines()]    #Removing extra spaces.
        file_item.close()
        doc_vector = [0]*model_dim
        vectors_found = 0
        words = []

        if n == 1:
            total_num_paragraphs += len(paragraphs)

        for paragraph in paragraphs:
            sentences = nltk.sent_tokenize(paragraph, nltk_language)    #Identifying sentences.

            if n == 1:
                total_num_sentences += len(sentences)

            for sentence in sentences:
                if args.tokenize:
                    tokens = nltk.tokenize.word_tokenize(sentence)    #Works well for many European languages.
                else:
                    tokens = sentence.split()

                if args.ignore_case:
                    tokens = [t.lower() for t in tokens]

                if args.validate_words:
                    allowed_tokens = [t for t in tokens if isword(t) and t not in stoplist]    #Filter allowed tokens.
                else:
                    allowed_tokens = [t for t in tokens if t not in stoplist]    #Filter allowed tokens.

                n_grams = nltk.everygrams(allowed_tokens, max_len=n)    #Call n-grams combinations.

                for n_gram in n_grams:
                    new_word = "_".join(n_gram)

                    if new_word not in stoplist:
                        words.append(new_word)

        ########################################################################
        ### TASK 2 - TEXT REPRESENTATION                                     ###
        ########################################################################

        #Sum all vectors found:
        for word in words:
            if word in vectors:
                doc_vector = [sum(x) for x in zip(*[doc_vector, vectors[word]])]
                vectors_found += 1

        #Dividing (arithmetic mean) final vector:
        if vectors_found != 0:
            doc_vector = [x / vectors_found for x in doc_vector]

        ########################################################################


        ########################################################################
        ### OUTPUT (WRITING TEXT REPRESENTATION MATRIX)                      ###
        ########################################################################

        out_file.write( "\t".join(str(e) for e in doc_vector) + "\t" + filepath.split('/')[-2].strip() + "\n" )

        ########################################################################

        filepath_i += 1
        end = time.time()
        eta = (total_operations-filepath_i)*(end-start)
        print_progress(filepath_i, total_operations, eta)

    out_file.close()

operation_end = time.time()
eta = operation_end-operation_start
print_progress(total_operations, total_operations, eta, final=True)
print("..................................................\n\n\n")

################################################################################

#Comparing output files:
if args.n_gram > 1:
    print("> Removing duplicated files:")
    print("..................................................")
    any_file_removed = False

    for n in reversed(range(2, args.n_gram+1)):
        if (filecmp.cmp(out_string + str(n), out_string + str(n-1), shallow=False)):
            any_file_removed = True
            os.remove(out_string + str(n))
            print(out_string + str(n) + " \t\t\t--> REMOVED")

    if not any_file_removed:
        print("- All files are different!")

    print("..................................................\n\n\n")

################################################################################


################################################################################

total_end = time.time()
time = format_time(total_end-total_start)
files = str(total_num_examples)
paragraphs = str(total_num_paragraphs)
sentences = str(total_num_sentences)
print("> Log:")
print("..................................................")
print("- Time: " + time)
print("- Input files: " + files)
print("- Input paragraphs: " + paragraphs)
print("- Input sentences: " + sentences)
print("..................................................\n")
log.write("\n\n> Log:\n")
log.write("\t- Time:\t\t\t" + time + "\n")
log.write("\t- Input files:\t\t" + files + "\n")
log.write("\t- Input paragraphs:\t\t" + paragraphs + "\n")
log.write("\t- Input sentences:\t\t" + sentences + "\n")
log.close()
