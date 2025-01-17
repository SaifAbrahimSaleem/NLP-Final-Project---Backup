from sklearn.svm import LinearSVC
from nltk.classify import SklearnClassifier
from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk import pos_tag, Tree, ngrams
from sklearn.model_selection import cross_val_score
from sklearn import metrics
from sklearn.metrics import precision_recall_fscore_support
from sklearn.feature_extraction.text import CountVectorizer
from random import shuffle
from collections import Counter, defaultdict
import spacy
from spacy import displacy
import numpy as np
import nltk
import csv
import re
import string
import math

nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
mode = ""
option = "0"
response = "0"
number_of_labels = 0
raw_data = [] # 2D Array Containing sentences
parsed_data = [] # Array Containing Tuples of sentences and labels
word_frequency = Counter()
word_label_occurrences = Counter()
preprocessed_data_words = []
preprocessed_data_grammars = []
labels = []
label_word_dict = defaultdict(list)
label_grammar_dict = defaultdict(list)
features = []

"""
    CHOICE METHODS
        Simple methods used to take input from a user and also validate input for the right choice
"""
def get_option():
    print(" ")
    print("Options: ")
    print("1: Classification using Training data ")
    print("2: Classification using Testing data")
    print("3: Exit")
    print(" ")
    choice = int(input())
    if choice not in [1,2,3]:
        print("Please only input a value between 1 and 3!")
        return get_option()
    else:
        return choice

def get_response():
    print(" ")
    print("1: Predict Gender")
    print("2: Predict Speaker")
    print(" ")
    choice = int(input())
    if choice not in [1,2]:
        print("Please only input 1 or 2!")
        return get_response()
    else:
        return choice

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
    print(" 3 - Predicting " + mode + " using more than one technique")
    print(" 4 - Predicting " + mode + " using both techniques")
    print("")
    choice = int(input())
    if choice not in [1,2,3,4]:
        print("select a number between 1 and 4 only Please!")
        return feature_choices()
    if choice == 6:
        return choose_multiple()
    else:
        return choice

def get_grammar_option():
    print("Choose from the list below: ")
    print(" 1 - Label Specific Grammars using Parts Of Speech (POS) Tagging")
    print(" 2 - Label Specific Grammars using Dependency Relations")
    print(" 3 - Label Specific Grammars using both Parts Of Speech (POS) Tagging and Dependency Relations")
    choice = int(input())
    if choice not in [1,2,3]:
        print("Please Choose a Value between 1 and 3!")
        get_grammar_option()
    else:
        return choice

def choose_n_grams():
    print("Enter the value of N for an N-Gram model (1-3): ")
    print("")
    choice = int(input())
    print("")
    if choice not in [1,2,3]:
        print("Please choose a value between 1 and 3!")
        choose_n_grams()
    else:
        return choice
"""##########################################################################################################"""

def print_vals(is_testing, mode):
    """         PRINTING THE MODE CHOSEN        """
    if is_testing:
        is_testing_string = "TESTING"
    else:
        is_testing_string = "TRAINING"
    return "Building a Classifier to predict the " + mode + " using " + is_testing_string + " data."

"""     SIMPLE PROGRESS BAR METHOD USED FOR TESTING PUROSES         """
def progress_bar (current_iteration, total_iterations, prefix = ''):
    completion = ("{0:.1f}").format(100 * (current_iteration / float(total_iterations)))
    progress = int(100 * current_iteration // total_iterations)
    bar = '█' * progress + '-' * (100 - progress)
    print('\r%s |%s| %s%% ' % (prefix, bar, completion), end="\r")
    if current_iteration == total_iterations:
        print()

def parse_line(line,mode):
    """
        PARSING EACH LINE
            lOGIC:
                - If the mode chosen is to predict the gender of the speaker, return the first and last columns of the line
                - Otherwise, the mode chosen is predicting the speaker to which, return the first and second columns of the line
    """
    if mode == "GENDER":
        return (line[0], line[2])
    else:
        return (line[0], line[1])

def preprocess_text(preprocessed_dataset, sentence, label, option, n_gram_choice):
    """
        PREPROCESSING TEXT FOR WORDS (SAME METHOD AS USED IN LAB 2 SOLUTIONS)
        STEPS:
            - Tokenize each sentence and remove punctuation (using regular expressions)
            - Change all characters in a word to lowercase
            - Remove all stopwords from tokens
            - Lemmatize words (use the lemma form of a group of words where possible)
            - append the result to preprocessed data (stored as a global variable)
    """
    regex_tokeniser = RegexpTokenizer(r'\w+')
    sentence_tokens = regex_tokeniser.tokenize(sentence)
    sentence_tokens = [token.lower() for token in sentence_tokens]
    sentence_tokens = [token for token in sentence_tokens if token not in set(stopwords.words('english'))]
    if option == 1:
        POS_Tagged_sentence_tokens = nltk.pos_tag(sentence_tokens)
        for index, token in enumerate(sentence_tokens):
            if len(sentence_tokens) != 0:
                sentence_tokens[index] += " " + str(POS_Tagged_sentence_tokens[index][1])
    elif option == 2:
        spacy_model = spacy.load('en_core_web_sm')
        dependency_relation = spacy_model(sentence)
        for index, token in enumerate(sentence_tokens):
            if len(sentence_tokens) != 0:
                if sentence_tokens[index] == dependency_relation[index].text:
                    sentence_tokens[index] += " " + str(dependency_relation[index].dep_)
    elif option == 3:
        spacy_model = spacy.load('en_core_web_sm')
        POS_Tagged_sentence_tokens = nltk.pos_tag(sentence_tokens)
        for index, token in enumerate(sentence_tokens):
            dependency_relation = spacy_model(sentence)
            if len(sentence_tokens) != 0:
                if token == dependency_relation[index].text:
                    sentence_tokens[index] += " " + str(POS_Tagged_sentence_tokens[index][1]) + " " + str(dependency_relation[index].dep_)
    lemmatiser = WordNetLemmatizer()
    sentence_tokens = [lemmatiser.lemmatize(token) for token in sentence_tokens]
    if n_gram_choice in [1,2,3]:
        sentence_tokens = ngrams(sentence_tokens, n_gram_choice)
    preprocessed_dataset.append((list(map(str,sentence_tokens)), label))

def clean_data(dataset):
    """
        REMOVING TUPLES WITH EMPTY LISTS FROM THE PREPROCESSED DATA
            After printing the preprocessed data, it was apparent that some tuples contained empty lists.
            Below is a method which was used to remove the tuples containing empty lists.
    """
    print("")
    count = 0
    num_nulls_ac = 0
    empty_index = []
    for index, entry in enumerate(dataset):
        if len(entry[0]) == 0:
            empty_index.append(index)
        count += 1
        progress_bar(count, len(dataset), "Cleaning Data: ")
    for index in sorted(empty_index, reverse=True):
        del dataset[index]


def load_csv(file_path, mode):
    """   OPENING THE CSV FILE AND PARSING EACH LINE AS A TUPLE  """
    with open(file_path, 'r', encoding = "utf8") as file:
        dataset_lines = csv.reader(file, delimiter = ',')
        print("Parsing Data: ")
        for l in dataset_lines:
            (line,label) = parse_line(l,mode)
            if label not in labels:
                labels.append(label)
            parsed_data.append((line,label))
            raw_data.append(l)

def label_unique_words():
    return None

def label_specific_words():
    """
        FINDING LABEL SPECIFIC WORDS
            In order to find the frequency of particular words, and the number of occurrences of a word in relation to a specific label,
            a simple method was created which will find one occurrence of each word for each label. This is to avoid any duplications later,
            when turning words into weights.
            STEPS:
                - Iterate the preprocessed data finding each entry
                - Then, for each word contained in the first position of an entry, check if the word is not contained in the labelled word dictionary
                - If a word is not in the labelled word dictionary, append the word as a value, using the label as a key
                - return the result
    """
    count = 0
    for entry in preprocessed_data_words:
        for word in entry[0]:
            if word not in label_word_dict[entry[1]]:
                label_word_dict[entry[1]].append(word)
        progress_bar(count, len(preprocessed_data_words), "Finding all label specific words: ")
        count+=1
    return label_word_dict

def label_specific_grammars():
    """GRAMMATICAL STRUCTURES"""
    count = 0
    for entry in preprocessed_data_grammars:
        for grammar in entry[0]:
            if grammar not in label_word_dict[entry[1]]:
                label_grammar_dict[entry[1]].append(grammar)
        progress_bar(count, len(preprocessed_data_grammars), "\r Finding all label specific words: ")
        count+=1
    return label_grammar_dict

def run_all():
    print("run_all")
    return None

def features_to_vector(dataset, features, option):
    print("")
    WF = 0
    ILF = 0
    iteration = 0
    feature_vectors = []
    if isinstance(features, defaultdict):
        """
            In this method of calculating weights, we use a slight variation of TF-IDF. As TD-IDF refers to the importance of a particular term
            appearing within a document where there is only one document with 2-18 labels, in the context of label specific words, we will be using
            WF-ILF (Word Frequency - Inverse Label Frequency) where the goal is to be able to note the importance of a particular word to a specific
            label. The calculation of WF-ILF follows the same logic of TF-IDF however, instead of refering to the number of documents, we refer to
            the number of labels (classes) when evaluating the importance of a particular word.
            STEPS:
                - iterate over the keys of 'features', which contains one occurrence of a word (to avoid overincrementing occurrences), specific to a label
                - Incriment the occurrences of a word contained within the features list (features[key]) by 1
                - If the word has not been added to the number of occurrences, then add it to the counter and assign it the value of 1
                - Once the number of word occurrences specific to each label is complete, iterate over the preprocessed data and compute the weight of
                  each word by calculating WF-ILF
                - Add the result to a dictionary to store the words and their weights
                - Append the result to a feature vector list, including the label (as a tuple)
                - Return the result
        """
        for key in list(features):
            for word in features[key]:
                if word_label_occurrences[word] <= len(labels):
                    try:
                        word_label_occurrences[word] += 1.0
                    except:
                        word_label_occurrences[word] = 1.0
        for entry in dataset:
            weighted_w_dict = {}
            for word in entry[0]:
                WF = word_frequency[word]/len(dataset)
                ILF = math.log(float(len(labels))/word_label_occurrences[word])
                word_weight = WF * ILF
                try:
                    weighted_w_dict[word] += word_weight
                except:
                    weighted_w_dict[word] = word_weight
            feature_vectors.append((weighted_w_dict,entry[1]))
            progress_bar(iteration, len(dataset), "\r Calculating weights: ")
            iteration+=1
        print("")
        return feature_vectors
    elif isinstance(features, list):
        print("list of features")
    # DO DA CALCULATIONZ
    return None

def select_features(features):
    if isinstance(features,list):
        for feature in features:
            if feature == 1:
                label_specific_word_dict = label_specific_words()
                # turn into weighted features
                # append to features
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
            iteration = 0
            n_gram_choice = choose_n_grams()
            for index, entry in enumerate(parsed_data):
                preprocess_text(preprocessed_data_words, entry[0], entry[1],0,n_gram_choice)
                progress_bar(index, len(parsed_data), "\r Pre-Processing Data: ")
                iteration += 1
            clean_data(preprocessed_data_words)
            iteration = 0
            for entry in preprocessed_data_words:
                for word in entry[0]:
                    try:
                        word_frequency[word] += 1.0
                    except:
                        word_frequency[word] = 1.0
                progress_bar(iteration, len(preprocessed_data_words), "\r Counting word frequency: ")
                iteration+=1
            print("")
            label_specific_word_dict = label_specific_words()
            weighted_list = features_to_vector(preprocessed_data_words, label_specific_word_dict,1)
            return weighted_list

        elif features == 2:
            iteration = 0
            option = get_grammar_option()
            n_gram_choice = choose_n_grams()
            for index, entry in enumerate(parsed_data):
                preprocess_text(preprocessed_data_grammars, entry[0], entry[1], option, n_gram_choice)
                progress_bar(index, len(parsed_data), "\r Pre-Processing Data: ")
                iteration += 1
            print("")
            iteration = 0
            for entry in preprocessed_data_grammars:
                for grammar in entry[0]:
                    try:
                        word_frequency[grammar] += 1.0
                    except:
                        word_frequency[grammar] = 1.0
                progress_bar(iteration, len(preprocessed_data_grammars), "\r Counting word frequency: ")
                iteration+=1
            clean_data(preprocessed_data_grammars)
            label_specific_grammar_list = label_specific_grammars()
            weighted_list = features_to_vector(preprocessed_data_grammars, label_specific_grammar_list, 2)
            return weighted_list

        elif features == 3:
            label_specific_sentiments()
            return

        elif features == 4:
            label_specific_semantics()
            return

        elif features == 5:
            sequence_classification()
            return

        elif features == 7:
            run_all()
            return

def cross_validate():
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
    features = feature_choices()
    number_of_labels = int(len(labels))
    iteration = 0
    weighted_data =  select_features(features)
    print("Training Classifier: ")
    classifier = SklearnClassifier(LinearSVC(loss='squared_hinge', max_iter=999999)).train(weighted_data)

    # make_predictions()
    return None


"""             PROGRAM UI              """
option = get_option()
if option == 1:
    """         TRAINING THE MODEL       """
    is_testing = False
    response = get_response()
    if response == 1:
        """         PREDICTING THE GENDER OF A SPEAKER GIVEN THE SENTENCES UTTERED          """
        mode = "GENDER"
        run_program(is_testing, mode)
    elif response == 2:
        """         PREDICTING THE SPEAKER GIVEN THE SENTENCES UTTERED          """
        mode = "SPEAKER"
        run_program(is_testing, mode)
    elif response == 3:
        """ EXIT THE PROGRAM """
        print("Exiting")
        sys.exit()
    else:
        print("Input numbers from 1-3 only Please!")
elif option == 2:
    """         TESTING THE MODEL       """
    is_testing = True
    response = get_response()
    if response == 1:
        """         PREDICTING THE GENDER OF A SPEAKER GIVEN THE SENTENCES UTTERED          """
        mode = "GENDER"
        run_program(is_testing, mode)
    elif response == 2:
        """         PREDICTING THE SPEAKER GIVEN THE SENTENCES UTTERED          """
        mode = "SPEAKER"
        run_program(is_testing, mode)
    elif response == 3:
        """ EXIT THE PROGRAM """
        print("Exiting")
        sys.exit()
    else:
        print("numbers from 1-3 only Please!")
elif option == 3:
    """ EXIT THE PROGRAM """
    print("Exiting")
    sys.exit()

"""
STEPS: --> PREPROCESS DATA --> FIND OUT WHAT FEATURES YOU WANT TO EVALUATE (FEATURE EXTRACTION & SELECTION) --> - CLASSIFY THE OUTPUT
"""
