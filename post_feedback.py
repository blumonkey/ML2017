
# coding: utf-8
"""

Author: Samuel Suraj Bushi
M.Sc, Univeristy of Alberta.

In this project, we try to predict the sentiment on reviews on Amazon, IMDB and Yelp!
We compare Lexicon-based, SVMs and Neural Network methods and use accuracy as a performance metric.
We report confusion matrices and micro-averaged precision recall and use statistical significance test
to test if the findings are statistically significant.

-- Project completed as part of Machine Learning Course (2017), under Prof. Martha White.

December 8th, 2017

"""

from __future__ import division
import numpy as np
import pandas as pd
import nltk
# from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem import SnowballStemmer
from nltk import word_tokenize
from nltk.tokenize import sent_tokenize
import re
from sklearn.neural_network import MLPClassifier
from sklearn import svm
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from scipy import stats
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from sklearn.grid_search import GridSearchCV

import urllib
import os
import zipfile

# Stopwords to be used in preprocessor
# stop = stopwords.words('english')
stop_edit = [u'i', u'me', u'my', u'myself', u'we', u'our', u'ours', u'ourselves', u'you', u'your', u'yours', u'yourself', u'yourselves', u'he', u'him', u'his', u'himself', u'she', u'her', u'hers', u'herself', u'it', u'its', u'itself', u'they', u'them', u'their', u'theirs', u'themselves', u'what', u'which', u'who', u'whom', u'this', u'that', u'these', u'those', u'am', u'is', u'are', u'was', u'were', u'be', u'been', u'being', u'have', u'has', u'had', u'having', u'do', u'does', u'did', u'doing', u'a', u'an', u'the', u'and', u'but', u'if', u'or', u'because', u'as', u'until', u'while', u'of', u'at', u'by', u'for', u'with', u'about', u'between', u'into', u'through', u'during', u'before', u'after', u'above', u'below', u'to', u'from', u'up', u'down', u'in', u'out', u'on', u'off', u'over', u'under', u'again', u'further', u'then', u'once', u'here', u'there', u'when', u'where', u'why', u'how', u'all', u'any', u'both', u'each', u'few', u'more', u'most', u'other', u'some', u'such', u'only', u'own', u'same', u'so', u'than', u'too', u'very', u'can', u'will', u'just', u'should', u'now']

def load_reviews():
    """
    Function that loads the reviews from data.txt.
	"""
    df = pd.read_csv('./data/data.txt', delimiter='\t', header=None)
    df.columns = ['Text', 'Sentiment']
    # return df.sample(frac=1)
    # return np.split(df.sample(frac=1), [int(.6*len(df)), int(.8*len(df))])
    train=df.sample(frac=0.8)
    test=df.drop(train.index)
    return train, test

def lemmatize(s):
    """
    Function that lemmatizes a given token, based on a heuristic of shortest word
    """
    lemmatizer = WordNetLemmatizer()
    a = lemmatizer.lemmatize(s, pos='n')
    b = lemmatizer.lemmatize(s, pos='v')
    if len(a) <= len(b):
        return a
    return b


def compress_word(s):
    """
    Function that compresses stressed words like 'coooooooooooooollllllllllllll' and 'wwwwwwwwaaaaaaaaaayyyyyyy'
    """
    patterns = [r'\bw+a+y+\b', r'\bc+o+o+l+\b', r'\bn+i+c+e+\b']  # Needs to be improved
    matches = ['way', 'cool', 'nice']
    for i in range(len(patterns)):
        p = patterns[i]
        m = matches[i]
        rx = re.compile(p)
        if re.match(p, s):
            return matches[i]
    return s


def preprocessor(text):
    """
    Function that preprocesses the input string, based on some heuristics.
    """
    regex = re.compile('\.{2,}')
    regex2 = re.compile(r'(\w+)\.(\w+)')
    sentences = sent_tokenize(text)
    total = []
    for s in sentences:
        txt = regex2.sub(r'\1. \2', s)																			# Add space after fullstops.
        tokens = word_tokenize(txt)
        tokens = [s.lower() for s in tokens]                                                                    # Convert to lowercase
        tokens = [unicode(x, errors='ignore') for x in tokens]                                                  # Convert to unicode
        tokens = [regex.sub('', x) for x in tokens]                                                             # Remove elipses
        tokens = [x for x in tokens if x not in '.,;!?']                                                        # Remove punctuations
        tokens = [x for x in tokens if len(x)!=0]                                                               # Remove empty tokens
        tokens = [x if x != "n't" else "not" for x in tokens]                                                   # Replace n't, 's and 'd with not, is and would
        tokens = [x if x != "'s" else "is" for x in tokens]
        tokens = [x if x != "'d" else "would" for x in tokens]
        tokens = map(lemmatize, tokens)                                                                         # Lemmatize the words
        tokens = map(compress_word, tokens)                                                                     # Compress possible words
        tokens = [x for x in tokens if x not in stop_edit]														# Remove stop-words
        total = total + tokens
        total.append("<TERM>")																					# Terminate sentence
    return total


def batch_preprocessor(inp):
    data_ = []
    """
    Applies the preprocessor on a dataframe and returns a dataframe of Text and Sentiment
    """
    for index, row in inp.iterrows():
        total = preprocessor(row['Text'])
        total = ' '.join(total)
        data_.append((total, row['Sentiment']))
    result = pd.DataFrame(data_)
    result.columns = ['Text', 'Sentiment']
    return result


class LexiconBasedSA():
	"""Implementation of Lexicon Based approach for Sentiment Analysis"""
	def __init__(self):
		self.vocab = {}

	def get_params(self, deep):
		"""
		LexiconBasedSA has no parameters, therefore we return an empty dictionary
		"""
		return {}

	def normalize(self):
		"""
		Function that normalizes the vocabulary
		"""
		for k in self.vocab.keys():
		    total = np.sum(self.vocab[k])
		    self.vocab[k] =  [x / (1.0*total) for x in self.vocab[k]]

	def fit(self, Xtrain, ytrain):
	    
	    """
	    Function that creates a lexicon from the training data
	    """
	    self.vocab = {}
	    train_data = pd.DataFrame(np.column_stack((Xtrain, ytrain)))
	    train_data.columns = ['Text', 'Sentiment']
	    curated = batch_preprocessor(train_data)
	    curated.columns = ['Text', 'Sentiment']

	    excl = ['<TERM>', '<END>']
	    for index, row in curated.iterrows():
	        text = row['Text']
	        words = text.split()
	        for word in words:
	            if word not in excl:
	                if word in self.vocab:
	                    self.vocab[word][row['Sentiment']] += 1
	                else:
	                    self.vocab[word] = [0, 0]
	                    self.vocab[word][row['Sentiment']] += 1
	    self.normalize()

	def predict(self, Xtest):
	    """
	    Function that predicts for new samples, from the sum of the vocab vectors
	    """
	    Xtest = pd.DataFrame(Xtest)
	    Xtest.columns = ['Text']
	    ytest = []
	    for index, row in Xtest.iterrows():
		    c_sample = preprocessor(row['Text'])
		    curr_out = np.array([0, 0])
		    for w in c_sample:
		        if w in self.vocab:
		            curr_out = curr_out + np.array(self.vocab[w])
		    tot = np.sum(curr_out)
		    curr_out = curr_out / float(tot)
		    ytest.append(np.argmax(curr_out))
		
	    return np.array(ytest)

def load_word2vec():
	"""
	Function that loads the word embeddings
	"""

	fname = './data/glove.6B.50d.txt'
	zip_path = "./data/glove.zip"

	# Download if not available
	if not os.path.isfile(fname): 
		print 'Can\'t find word vectors. Downloading now...'
		opener = urllib.URLopener()
		opener.retrieve(" https://nlp.stanford.edu/data/glove.6B.zip", zip_path)

		# Unzip downloaded file
		zip_ref = zipfile.ZipFile(zip_path, 'r')
		zip_ref.extract('glove.6B.50d.txt', './data/')
		zip_ref.close()

		# Remove zip file
		os.remove(zip_path)

	word2vec = {}
	with open(fname, 'r') as f:
	    for line in f:
	        tabs = line.split(' ', 1)
	        word2vec[tabs[0]] = np.array([float(x) for x in tabs[1].split(' ')])
	return word2vec


def list2vec(tokens, word2vec):
    """
	Function that converts a list of tokens to bag of words representation in word2vec
	"""
    out = np.zeros(50)
    for t in tokens:
        if t in word2vec:
            out += word2vec[t]
    return out

def normalized(a, axis=-1, order=2):
    """
	Function that normnalizes vectors
	"""
    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2==0] = 1
    return a / np.expand_dims(l2, axis)


def batch_preprocessor_w2v(inp, w2v):
    """
	Batch preprocessor for word2vec
	"""
    data_ = []
    for index, row in inp.iterrows():
        total = preprocessor(row['Text'])
        # total = normalized(list2vec(total))[0]
        total = list2vec(total, w2v)
        data_.append((total, row['Sentiment']))
    
    result = pd.DataFrame(data_)
    result.columns = ['Text', 'Sentiment']
    return result

# Chosen for optimal running time
num_runs = 5


# Storing the errors and confusion matrices for each model over the runs.

errors = {
	'lbsa': [],
	'nn': [],
	'svm': []
}

confusion_matrices = {
	'lbsa': [],
	'nn': [],
	'svm': []
}

# Parameter pool to optimize upon, for NN and SVMs.
MLP_parameters = {
	'solver':['adam', 'sgd'], 
	'alpha':[1e-3, 1e-4, 1e-5],                                # * or / 10 on the default value used by sklearn
	'hidden_layer_sizes': [(4), (8), (16), (32)] 
}

SVM_parameters = {
	'kernel': ['linear'], 
	'C':[1.0, 2.0, 2.0**2, 2.0**3, 2.0**4, 2.0**5]             # We dont check C values less than 1, as they perform worse than the rest, and add to the expense of unnecessary computation.
}

def main():
	# Independent runs
	for i in range(num_runs):
		
		# Loading the resources
	    reviews_train, reviews_test = load_reviews()
	    word2vec = load_word2vec()
	    reviews_train_w2v = batch_preprocessor_w2v(reviews_train, word2vec)
	    reviews_test_w2v = batch_preprocessor_w2v(reviews_test, word2vec)


	    # Running k-fold cross validation with k = 5

	    # For LexiconBasedSA
	    print 'Lexicon Based'
	    clf = LexiconBasedSA()
	    X = reviews_train['Text']
	    y = reviews_train['Sentiment']

	    clf.fit(X, y)
	    y_true, y_pred_lex = reviews_test['Sentiment'], clf.predict(reviews_test['Text'])

	    # Reporting the confusion matrix
	    print 'Confusion Matrix'
	    cf = confusion_matrix(y_pred_lex, y_true)
	    print cf
	    confusion_matrices['lbsa'].append(cf)

	    # Reporting the accuracy
	    print("Accuracy: %f" % (np.sum(y_pred_lex==y_true)/float(len(reviews_test))))
	    errors['lbsa'].append(np.sum(y_pred_lex==y_true)/float(len(reviews_test)))
	    

	    # For Neural Networks
	    print 'Neural Networks:'
	    # Internal cross-validation using Grid Search
	    clf = GridSearchCV(MLPClassifier(), MLP_parameters, cv=5, scoring='accuracy')
	    X = reviews_train_w2v['Text']
	    y = reviews_train_w2v['Sentiment']
	    clf.fit(np.array(list(X)), y)

	    print "Best parameters set found on development set:"
	    print ''
	    print clf.best_estimator_

	    y_true, y_pred_nn = reviews_test_w2v['Sentiment'], clf.predict(np.array(list(reviews_test_w2v['Text'])))

	    print 'Confusion Matrix'
	    cf = confusion_matrix(y_pred_nn, y_true)
	    print cf
	    confusion_matrices['nn'].append(cf)

	    print("Accuracy: %f" % (np.sum(y_pred_nn==y_true)/float(len(reviews_test))))
	    errors['nn'].append((np.sum(y_pred_nn==y_true)/float(len(reviews_test))))


	    # For SVMs
	    print 'SVM:'

	    clf = GridSearchCV(svm.SVC(), SVM_parameters, cv=5, scoring='accuracy')
	    X = reviews_train_w2v['Text']
	    y = reviews_train_w2v['Sentiment']
	    clf.fit(np.array(list(X)), y)

	    print "Best parameters set found on development set:"
	    print ''
	    print clf.best_estimator_

	    y_true, y_pred_svm = reviews_test_w2v['Sentiment'], clf.predict(np.array(list(reviews_test_w2v['Text'])))

	    print 'Confusion Matrix'
	    cf = confusion_matrix(y_pred_svm, y_true)
	    print cf
	    confusion_matrices['svm'].append(cf)

	    print("Accuracy: %f" % (np.sum(y_pred_svm==y_true)/float(len(reviews_test))))
	    errors['svm'].append((np.sum(y_pred_svm==y_true)/float(len(reviews_test))))


	# Doing statistical significance tests.

	print 'Performing statistical tests (alpha = 0.05)...'
	print ''
	print 'LexiconBasedSA vs Neural Network:'

	statistic, pvalue =  stats.ttest_ind(errors['lbsa'], errors['nn'], equal_var=False)

	print 'One tailed p-value: %f ' % (pvalue / 2.0)
	print 'One tailed t-stat: %f ' % statistic

	print ''
	print 'Neural Network vs SVM:'

	statistic, pvalue =  stats.ttest_ind(errors['nn'], errors['svm'], equal_var=False)

	print 'One tailed p-value: %f ' % (pvalue / 2.0)
	print 'One tailed t-stat: %f ' % statistic

	print ''
	print 'SVM vs LexiconBasedSA:'

	statistic, pvalue =  stats.ttest_ind(errors['svm'], errors['lbsa'], equal_var=False)

	print 'One tailed p-value: %f ' % (pvalue / 2.0)
	print 'One tailed t-stat: %f ' % statistic

if __name__=="__main__":
	main()