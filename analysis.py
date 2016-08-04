# Region Classification

import random
from collections import defaultdict
from gensim import models, matutils
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import train_test_split, KFold
from sklearn.metrics import accuracy_score, classification_report, \
confusion_matrix, f1_score
#from sklearn.metrics import roc_auc_score, roc_curve, auc, precision_recall_curve,\
#precision_recall_fscore_support


def sample_input(filename, n):
    text_list = []
    labels = []
    # The total number of sequences is 323101 (e.g., wc -l regions.txt)
    indices = set(random.sample(range(323101),n))
    counter = 0
    with open(filename) as fp:
        for line in fp:
            if counter in indices:
                data = line.split()
                text_list.append(data[0])
                labels.append(data[1])
            counter += 1
    return (text_list, labels)
    

def process_text(text_list, min_word_length, max_word_length, max_text_length):
    docs_list = []
    set_of_keys = set()
    for i in range(len(text_list)):
        text = text_list[i]
        text_dict = defaultdict(int)
        for width in range(min_word_length, max_word_length+1):
            for j in range(max_text_length+1-width):
                text_dict[text[j:j+width]] += 1
                set_of_keys.add(text[j:j+width])
        docs_list.append(text_dict)
    return (docs_list, set_of_keys)


def keytoindex(set_of_keys):
    key_to_ind = {}
    ind_to_key = {}
    i = 0
    for elem in set_of_keys:
        key_to_ind[elem] = i
        ind_to_key[i] = elem
        i += 1
    return (key_to_ind, ind_to_key)
    
    
def get_corpus(docs_list, key_to_ind):
    list_of_lists_of_tuples = []
    for doc in docs_list:
        list_of_tuples = []
        for key in doc:
            index = key_to_ind[key]
            value = doc[key]
            list_of_tuples.append((index,value))
        list_of_lists_of_tuples.append(list_of_tuples)
    return list_of_lists_of_tuples
    

def get_Xy(corpus, labels, ndims):
    # Term frequency inverse document frequency (tfidf) weighting:
    # reflects how important a word is to a document in a corpus
	tfidf = models.TfidfModel(corpus)
	tfidf_corpus = tfidf[corpus]	    
	docs_tfidf = [doc for doc in tfidf_corpus]
	##scipy_csc_matrix = matutils.corpus2csc(tfidf_corpus).toarray().transpose()	
	# Builds an LSI space from the input TFIDF matrix, it uses SVD
	# for dimensionality reduction with num_topics = dimensions
	lsi = models.LsiModel(tfidf_corpus, num_topics = ndims)
	lsi_corpus = lsi[tfidf_corpus]
	docs_lsi = [doc for doc in lsi_corpus]
	X = matutils.corpus2dense(lsi_corpus, num_terms = ndims).transpose()
	# Convert labels to: promoter: 0, enhancer: 1
	y = []
	error_ind = []
	for i in range(len(labels)):
	    if labels[i] == 'promoter':
	        y.append(0)
	    elif labels[i] == 'enhancer':
	        y.append(1)
	    else:
	        print "Promoter / enhancer not recorded at index", i
	        error_ind.append(i)
	y = np.asarray(y)
	return (X, y)


def train_model(X, y):
    kf = KFold(len(y), n_folds=10)
    f1logr = 0.0
    f1dtc = 0.0
    f1rfc = 0.0
    for train_index, test_index in kf:
        Xtrain, Xvalid = X[train_index], X[test_index]
        ytrain, yvalid = y[train_index], y[test_index]
        ## Logistic Regression
        logr = LogisticRegression()
        modellogr = logr.fit(Xtrain, ytrain)
        ypred = logr.predict(Xvalid)
        f1logr += f1_score(yvalid, ypred)
        ## Decision Tree
        #dtc = DecisionTreeClassifier(class_weight='balanced')
        #modeldtc = dtc.fit(Xtrain, ytrain) 
        #ypred = dtc.predict(Xvalid)
        #f1dtc += f1_score(yvalid, ypred)
        ## Random Forest 
        #rfc = RandomForestClassifier()
        #modelrfc = rfc.fit(Xtrain, ytrain) 
        #ypred = rfc.predict(Xvalid)
        #f1rfc += f1_score(yvalid, ypred)
    f1logr = f1logr/10.0
    #f1dtc = f1dtc/10.0
    #f1rfc = f1rfc/10.0
    print 'Average F1 score: ', f1logr
    #print f1dtc
    #print f1rfc
    # Logistic Regression Stats
    Xtrain, Xvalid, ytrain, yvalid = train_test_split(X, y, test_size=0.4)
    logr = LogisticRegression()
    modellogr = logr.fit(Xtrain, ytrain)
    ypred = logr.predict(Xvalid)
    print 'One test results:'
    print 'Classification Report:\n', classification_report(yvalid, ypred)
    print 'Confusion Matrix:\n', confusion_matrix(yvalid, ypred)
    print 'F1 Score:', f1_score(yvalid, ypred)
    return


if __name__ == '__main__':
	path = "./"
	filename = path + "regions.txt"
	random.seed(1234)
	text_list, labels = sample_input(filename, 10000)
	min_word_length = 1
	max_word_length = 4
	max_text_length = 1000
	docs_list, set_of_keys = process_text(text_list, min_word_length, max_word_length, max_text_length)
	key_to_ind, ind_to_key = keytoindex(set_of_keys)
	corpus = get_corpus(docs_list, key_to_ind)
	X, y = get_Xy(corpus, labels, 100)
	train_model(X, y)
	

