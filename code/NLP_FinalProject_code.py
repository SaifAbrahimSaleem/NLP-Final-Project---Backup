from sklearn.svm import LinearSVC
from nltk.classify import SklearnClassifier
from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from sklearn.model_selection import cross_val_score
from sklearn import metrics
from sklearn.metrics import precision_recall_fscore_support
from sklearn.feature_extraction.text import CountVectorizer
from random import shuffle
from collections import Counter
import numpy as np
import nltk
import csv
import re
import string

nltk.download('wordnet')
nltk.download('stopwords')

mode = ""
option = "0"
response = "0"
raw_data = [] # 2D Array Containing sentences
parsed_data = [] # Array Containing Tuples of sentences and labels
preprocessed_data = []
features = {}

def get_option():
    print(" ")
    print("Options: ")
    print("1: Classification using Training data ")
    print("2: Classification using Testing data")
    print("3: Exit")
    print(" ")
    return input()

def get_response():
    print(" ")
    print("1: Predict Gender")
    print("2: Predict Speaker")
    print(" ")
    return input()

def choose_multiple():
    print("Input Choices (using numbers 1-5, each seperated by a space)")
    choices = []
    choices = input().split()
    choices = list(map(int, choices))
    for c in choices:
        if c not in [1,2,3,4,5]:
            print("select a number between 1 and 5 only Please!")
            return choose_multiple()
    return choices

def feature_choices():
    print("Choose from the list below: ")
    print(" 1 - Predicting " + mode + " using " + mode + " specific words")
    print(" 2 - Predicting " + mode + " using " + mode + " specific grammars")
    print(" 3 - Predicting " + mode + " using " + mode + " specific sentiments")
    print(" 4 - Predicting " + mode + " using " + mode + " the semantic content of utterances")
    print(" 5 - Predicting " + mode + " using sequence classification")
    print(" 6 - Predicting " + mode + " using more than one technique")
    print(" 7 - Predicting " + mode + " using all techniques")
    choice = int(input())
    if choice not in [1,2,3,4,5,6,7]:
        print("select a number between 1 and 7 only Please!")
        return feature_choices()
    if choice == 6:
        return choose_multiple()
    else:
        return choice

def print_vals(is_testing, mode):
    if is_testing:
        is_testing_string = "TESTING"
    else:
        is_testing_string = "TRAINING"
    return "Building a Classifier to predict the " + mode + " using " + is_testing_string + " data."

def progress_bar (current_iteration, total_iterations, prefix = ''):
    completion = ("{0:.1f}").format(100 * (current_iteration / float(total_iterations)))
    progress = int(100 * current_iteration // total_iterations)
    bar = '█' * progress + '-' * (100 - progress)
    print('\r%s |%s| %s%% ' % (prefix, bar, completion), end="\r")
    if current_iteration == total_iterations:
        print()

def parse_line(line,mode):
    if mode == "GENDER":
        return (line[0], line[2])
    else:
        return (line[0], line[1])

def preprocess_text(sentence, label):
        # Removing punctuation from sentences and tokenising the line
        regex_tokeniser = RegexpTokenizer(r'\w+')
        sentence_tokens = regex_tokeniser.tokenize(sentence)
        # changing the words in the line to lowercase
        sentence_tokens = [token.lower() for token in sentence_tokens]
        # removing stopwords
        sentence_tokens = [token for token in sentence_tokens if token not in set(stopwords.words('english'))]
        # lemmatising the line
        lemmatiser = WordNetLemmatizer()
        sentence_tokens = [lemmatiser.lemmatize(token) for token in sentence_tokens]
        preprocessed_data.append((sentence_tokens, label))

def clean_data():
    """     REMOVING TUPLES WITH EMPTY LISTS FROM THE DATA     """
    print("")
    print("Cleaning Data: ")
    count = 0
    num_nulls_ac = 0
    empty_index = []
    for index, entry in enumerate(preprocessed_data):
        if len(entry[0]) == 0:
            empty_index.append(index)
    for index in sorted(empty_index, reverse=True):
        del preprocessed_data[index]


def load_csv(file_path, mode):
    with open(file_path, 'r', encoding = "utf8") as file:
        dataset_lines = csv.reader(file, delimiter = ',')
        print("Parsing Data: ")
        for l in dataset_lines:
            (line,label) = parse_line(l,mode)
            parsed_data.append((line,label))
            raw_data.append(l)

def label_specific_words():
    print("label_specific_words")
    return None

def label_specific_grammars():
    print("label_specific_grammars")
    return None

def label_specific_sentiments():
    print("label_specific_sentiments")
    return None

def label_specific_semantics():
    print("label_specific_semantics")
    return None

def sequence_classification():
    print("sequence_classification")
    return None

def run_all():
    print("run_all")
    return None

def select_features():
    features = feature_choices()
    print(features)
    if isinstance(features,list):
        for feature in features:
            if feature == 1:
                label_specific_words()
            if feature == 2:
                label_specific_grammars()
            if feature == 3:
                label_specific_sentiments()
            if feature == 4:
                label_specific_semantics()
            if feature == 5:
                sequence_classification()
    elif isinstance(features,int):
        if features == 1:
            label_specific_words()
        elif features == 2:
            label_specific_grammars()
        elif features == 3:
            label_specific_sentiments()
        elif features == 4:
            label_specific_semantics()
        elif features == 5:
            sequence_classification()
        elif features == 7:
            run_all()

def cross_validate():
    return None

def build_classifier():
    return None

def get_vals():
    return None

def make_predictions():
    return None


"""             MAIN PROGRAM            """
def run_program(is_testing, mode):
    """########## CHECKING WHAT THE PROGRAM IS GOING TO EXECUTE ##########"""
    print(" ")
    print(print_vals(is_testing, mode))
    """###################################################################"""
    iteration = 0
    file_path = ''
    if is_testing:
        file_path = 'Data/datasets/test.csv'
    else:
        file_path = 'Data/datasets/training.csv'
    load_csv(file_path, mode)
    # for index, entry in enumerate(parsed_data):
    #     preprocess_text(entry[0], entry[1])
    #     progress_bar(index, len(parsed_data), "\r Pre-Processing Data: ")
    #     iteration += 1
    # print(preprocessed_data)
    # clean_data()
    select_features()
    # build_classifier()
    # make_predictions()
    return None


"""             PROGRAM UI              """
while int(option) != 1 or int(option) != 2 or int(option) != 3:
    option = get_option()
    if int(option) == 1:
        """         TRAINING THE MODEL       """
        is_testing = False
        while int(response) != 1 or int(response) != 2 or int(response) != 3:
            response = get_response()
            if int(response) == 1:
                """         PREDICTING THE GENDER OF A SPEAKER GIVEN THE SENTENCES UTTERED          """
                mode = "GENDER"
                run_program(is_testing, mode)
                break

            elif int(response) == 2:
                """         PREDICTING THE SPEAKER GIVEN THE SENTENCES UTTERED          """
                mode = "SPEAKER"
                run_program(is_testing, mode)
                break

            elif int(response) == 3:
                """ EXIT THE PROGRAM """
                print("Exiting")
                break

            else:
                print("Input numbers from 1-3 only Please!")
                continue
        break
    elif int(option) == 2:
        """         TESTING THE MODEL       """
        is_testing = True
        while int(response) != 1 or int(response) != 2 or int(response) != 3:
            response = get_response()
            if int(response) == 1:
                """         PREDICTING THE GENDER OF A SPEAKER GIVEN THE SENTENCES UTTERED          """
                mode = "GENDER"
                run_program(is_testing, mode)
                break

            elif int(response) == 2:
                """         PREDICTING THE SPEAKER GIVEN THE SENTENCES UTTERED          """
                mode = "SPEAKER"
                run_program(is_testing, mode)
                break

            elif int(response) == 3:
                """ EXIT THE PROGRAM """
                print("Exiting")
                break

            else:
                print("numbers from 1-3 only Please!")
                continue
        break

    elif int(option) == 3:
        """ EXIT THE PROGRAM """
        break

    else:
        print("Input numbers from 1-3 only Please!")
        continue

"""
STEPS: --> PREPROCESS DATA --> FIND OUT WHAT FEATURES YOU WANT TO EVALUATE (FEATURE EXTRACTION & SELECTION) --> - CLASSIFY THE OUTPUT
"""
