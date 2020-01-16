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

"""
CHOOSE THE PREDICTION SCENARIO:
    - Option 1 - Set the value of the "option" variable to 1 to predict the gender of the speaker
    - Option 2 - Set the value of the "option" variable to 2, in order to predict the character's/speaker's name
"""

########## GLOBAL VARIABLES #########
option = 1
label_val = 0
raw_train_data = [] # a list containing a tuple (raw line, label(either gender or character name)) for training data
raw_test_data = [] # a list containing a tuple (raw line, label(either gender or character name)) for test data
preprocessed_train_data = [] # a list containing a tuple (preprocessed line, label(either gender or character name)) for training data
preprocessed_test_data = [] # a list containing a tuple (raw line, label(either gender or character name)) for test data
feature_dict = {}
utterances = Counter()
# m_words, f_words = Counter(), Counter()
# unique_m_words = {}
# unique_f_words = {}
# unique_train_words = []
# unique_test_words = []

def load_csv(file_path, option, testing):
    """
        * The option chosen will determine the label *
        This method will:
            - Load the csv and store the contents as fields (containing the contents of each field: [line, character/speaker, gender])
            ** OPTION 1 **
                - Parse each line by extracting the line and the gender of each field
                - Append a tuple containing the raw line spoken and the gender of the speaker to the raw_data list
                - Append a tuple containing the preprocessed line (cleaned, lowered, tokenised and lemmatised) spoken and the gender
                  of the speaker to the preprocessed_train_data list to be later used for training the classifier
            ** OPTION 2 **
                - Parse each line by extracting the line and the characher/speaker name of each field
                - Append a tuple containing the raw line spoken and the name of the character/speaker to the raw_data list
                - Append a tuple containing the preprocessed line (cleaned, lowered, tokenised and lemmatised) spoken and the name
                  of the character/speaker to the preprocessed_train_data list to be later used for training the classifier
Needs to be cut down in new version
    """
    print("Loading data from file: ")
    if not testing:
        with open(file_path, 'r', encoding = "utf8") as train_file:
            fields = csv.reader(train_file, delimiter=',')
            if option == 1:
                label_val = 1
            elif option == 2:
                label_val = 2
            for line in fields:
                (line, label) = parseLine(line,label_val)
                raw_train_data.append((line,label))
                preprocessed_train_data.append((features_to_vector(preprocess(line)),label))
    else:
        with open(file_path, 'r', encoding = "utf8") as test_file:
            fields = csv.reader(test_file, delimiter=',')
            if option == 1 or option == 2:
                if option == 1:
                    label_val = 1
                elif option == 2:
                    label_val = 2
                for line in fields:
                    (line, label) = parseLine(line,label_val)
                    raw_test_data.append((line,label))
                    preprocessed_test_data.append((features_to_vector(preprocess(line)),label))
            else:
                print("ERROR! Wrong Option Value Chosen!")
                print("Options are:")
                print(" - 1: Gender Prediction")
                print(" - 2: Character/Speaker Prediction")

def parseLine(line, label_val):
    """
    * The option chosen will determine the label *
        Since option 1 denotes the classification task of predicting the gender of the speaker:
            - return a tuple containing the line spoken and the gender of the speaker
        otherwise (else):
            - return a tuple containing the line spoken and the name of the speaker
    """
    # print("Parsing corpus: ")
    if label_val == 1:
        return (line[0], line[2])
    elif label_val == 2:
        return (line[0], line[1])
    else:
        print("ERROR! something has gone wrong! check the code!")

def preprocess(line):
    """
        Cleaning and tokenising sentences:
            - Remove punctuation from sentences
            - convert all words to lowercase
            - Tokenise the sentence and return a list of tokenised sentences
            - remove stopwords from the tokenised sentence
            - lemmatising words in the sentence
        This method will return a list of tokens from one sentence passed to this method
may need to be cut down if possible
    """
    # print("Pre-Processing data: ")
    # Removing punctuation from sentences and tokenising the line
    regex_tokeniser = RegexpTokenizer(r'\w+')
    line_tokens = regex_tokeniser.tokenize(line)
    # changing the words in the line to lowercase
    line_tokens = [token.lower() for token in line_tokens]
    # removing stopwords
    line_tokens = [token for token in line_tokens if token not in set(stopwords.words('english'))]
    # lemmatising the line
    lemmatiser = WordNetLemmatizer()
    line_tokens = [lemmatiser.lemmatize(token) for token in line_tokens]
    return line_tokens

def select_features(val):
    """
    KEY: ("||" -> Incomplete, "|*|" -> Complete)
    This method serves as a feature selector for improving the accuracy of the model. Exploring the different options which
    pertain to sentences derived from the raw data and their structure. Below are the options which may be used for consideration:
        FEATURE OPTIONS: (options will be stored in an array hence the proceeding enumerations for each option)
            - 0: All words contained in the corpus and their weights |*|
            - 1: Unique words contained in the corpus and their weights |*|
            - 2: Punctuations used and their weights (This can be used in parallel to words) ||
            - 3: Grammatical structure and features of sentences ||
            - 4: Speech styles used by a character ||
            - 5: Semantic content of utterances ||
            - 6: Sequence classification exploration ||
        ***** NOTE: options 0 and 1 can be used as either 0 or 1 and not both *****
    """
    return None

def features_to_vector(words):
    """

        TO DO: Could convert features to vectors based on gender/name maybe???

    """
    counts = Counter(words) # Create a counter and pass it words
    return {word: counts[word]/sum(counts.values()) for word in counts.keys()} # Assigning weights to words

"""
    TO DO:
        CHANGE THIS METHOD TO FIT MY MODEL
"""
def crossValidate(dataset, folds):
    shuffle(dataset)
    results = []
    foldSize = int(len(dataset)/folds)
    for i in range(0,len(dataset),int(foldSize)):
        # insert code here that trains and tests on the 10 folds of data in the dataset
        print("Fold start on items %d - %d" % (i, i+foldSize))
        myTestData = dataset[i:i+foldSize]
        myTrainData = dataset[:i] + dataset[i+foldSize:]
        classifier = trainClassifier(myTrainData)
        y_true = map(lambda x: x[1], myTestData)
        y_pred = predictLabels(myTestData, classifier)
        results.append(precision_recall_fscore_support(y_true, y_pred, average='weighted'))
    avgResults = map(np.mean,zip(*results)[:3])
    return avgResults

def predictLabels(preprocessed_test_data, classifier):
    return classifier.classify_many(map(lambda t: t[0], preprocessed_test_data))

def trainClassifier(corpus):
    return SklearnClassifier(LinearSVC(loss='squared_hinge', max_iter=3000)).train(corpus)


####################### MAIN #######################
training_path = 'Data/training.csv'
test_path = 'Data/test.csv'
load_csv(training_path, option, testing=False)
print(raw_train_data)
print(preprocessed_train_data)
classifier = trainClassifier(unique_train_words)
load_csv(test_path, option, testing=True)
unique_words(preprocessed_test_data, testing=False)
test_true = list(map(lambda test: test[1], preprocessed_test_data))#####
test_predictions = predictLabels(preprocessed_test_data, classifier)
finalScores = precision_recall_fscore_support(test_true, test_predictions, average='weighted')
print("Done training!")
print("Precision: %f\nRecall: %f\nF Score:%f" % finalScores[:3])

"""
    TO DO:
        - SPLIT_DATA -> PERCENTAGE SPLIT
        - CROSS VALIDATION
        - FEATURE SELECTION (PUNCTUATION IN SENTENCES):
            * at the moment only looking at words and their weights *
        - SEMANTIC CONTENT (SENTIMENT ANALYSIS{
            * WHO IS MORE LIKELY TO SAY SOMETHING PERTAINING TO A SENTIMENT?
            * WHO IS MORE LIKELY TO USE CERTAIN SEMANTIC UTTERANCES
        })
        - {     CLASSIFIERS FOR PREDICTION ***| DONE |***    }
        - WORD2VEC
        - DEPENDENCY GRAMMARS
        - POS TAGGING
        - CRF/HMM TAGGER (TO BE EXPLORED)


    """
