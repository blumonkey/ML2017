
# coding: utf-8

from __future__ import division
import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem import SnowballStemmer
from nltk import word_tokenize
from nltk.tokenize import sent_tokenize
import re
from sklearn.neural_network import MLPClassifier
from sklearn import svm
from sklearn.model_selection import cross_val_score

# Stopwords to be used in preprocessor
stop = stopwords.words('english')
stop_edit = [u'i', u'me', u'my', u'myself', u'we', u'our', u'ours', u'ourselves', u'you', u'your', u'yours', u'yourself', u'yourselves', u'he', u'him', u'his', u'himself', u'she', u'her', u'hers', u'herself', u'it', u'its', u'itself', u'they', u'them', u'their', u'theirs', u'themselves', u'what', u'which', u'who', u'whom', u'this', u'that', u'these', u'those', u'am', u'is', u'are', u'was', u'were', u'be', u'been', u'being', u'have', u'has', u'had', u'having', u'do', u'does', u'did', u'doing', u'a', u'an', u'the', u'and', u'but', u'if', u'or', u'because', u'as', u'until', u'while', u'of', u'at', u'by', u'for', u'with', u'about', u'between', u'into', u'through', u'during', u'before', u'after', u'above', u'below', u'to', u'from', u'up', u'down', u'in', u'out', u'on', u'off', u'over', u'under', u'again', u'further', u'then', u'once', u'here', u'there', u'when', u'where', u'why', u'how', u'all', u'any', u'both', u'each', u'few', u'more', u'most', u'other', u'some', u'such', u'only', u'own', u'same', u'so', u'than', u'too', u'very', u'can', u'will', u'just', u'should', u'now']

def load_reviews():
    """
    Function that loads the reviews from data.txt.
	"""
    df = pd.read_csv('data/shuf_data.txt', delimiter='\t', header=None)
    df.columns = ['Text', 'Sentiment']
    return df
    # return np.split(df.sample(frac=1), [int(.6*len(df)), int(.8*len(df))])

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
	word2vec = {}
	with open('data/glove.6B.50d.txt', 'r') as f:
	    for line in f:
	        tabs = line.split(' ', 1)
	        word2vec[tabs[0]] = np.array([float(x) for x in tabs[1].split(' ')])
	return word2vec


def list2vec(tokens):
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


def batch_preprocessor_w2v(inp):
    """
	Batch preprocessor for word2vec
	"""
    data_ = []
    for index, row in inp.iterrows():
        total = preprocessor(row['Text'])
        # total = normalized(list2vec(total))[0]
        total = list2vec(total)
        data_.append((total, row['Sentiment']))
    
    result = pd.DataFrame(data_)
    result.columns = ['Text', 'Sentiment']
    return result

# Loading the resources
reviews = load_reviews()
word2vec = load_word2vec()
reviews_w2v = batch_preprocessor_w2v(reviews)


# Running k-fold cross validation with k = 5

# For LexiconBasedSA
print 'Lexicon Based'
clf = LexiconBasedSA()
X = reviews['Text']
y = reviews['Sentiment']
scores = cross_val_score(clf, X, y, cv=5, scoring='accuracy')
print scores
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

# For Neural Networks
print 'Neural Networks:'

clf = MLPClassifier(solver='adam', alpha=1e-5, hidden_layer_sizes=(100, 50))
X = reviews_w2v['Text']
y = reviews_w2v['Sentiment']
scores = cross_val_score(clf, np.array(list(X)), y, cv=5, scoring='accuracy')
print scores
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))


# For SVMs
print 'SVM:'

clf = svm.SVC(C=5.0)
X = reviews_w2v['Text']
y = reviews_w2v['Sentiment']
scores = cross_val_score(clf, np.array(list(X)), y, cv=5, scoring='accuracy')
print scores
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))


# So the current rankings are:
# -----
#     - Lexicon based methods: Accuracy: 0.80 (+/- 0.04)
#     - SVMs: Accuracy: 0.78 (+/- 0.02)
#     - Neural Networks: Accuracy: 0.77 (+/- 0.04)
#     
# TODO:
# -----
#     - ROC?
#     - Confusion Matrix
