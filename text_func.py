import pandas as pd
import itertools
import networkx as nx
from itertools import combinations
import spacy
from nltk.stem import SnowballStemmer
import string
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import warnings
import scipy.sparse.linalg
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

from textblob import TextBlob
import numpy as np

def calculate_sentiment(text):
    blob = TextBlob(text)
    return blob.sentiment.polarity

def delete_rows_by_values(df, col, list_value):
    """
    Delete rows in a DataFrame where specific columns have certain values.

    Parameters:
    df (pd.DataFrame): The DataFrame to process.
    col (string): Name of column
    list_value (list): List of value that will be delete

    Returns:
    pd.DataFrame: The filtered DataFrame.
    """
    for val in list_value:
        df = df[df[col] != val]
    return df
def preprocessed_data(df, del_cols=None, target='label', extra_col='mostly-true'):
    if del_cols is None:
        del_cols = ['half-true', 'barely-true', 'pants-fire']

    df = delete_rows_by_values(df, target, del_cols)

    df.loc[df[target] == extra_col, target] = 'true'

    return df

def wordopt(text):
    print(text)
    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'\w*\d\w*', '', text)
    return text

def stemming(text, stemmer):

    tokens = text
    stemmed = [stemmer.stem(token) for token in tokens]
    return stemmed

def process_text(text, word_tokenize, stop_words, lemmatizer_):

    tokens = word_tokenize(text.lower())
    tokens = [word for word in tokens if word not in stop_words and word.isalpha()]
    lemmatized_tokens = [lemmatizer_.lemmatize(word) for word in tokens]

    return lemmatized_tokens

def sentence_processed(list_text):
    return " ".join(list_text)


def preprocess_natural_language(df, tokens='preprocess_tokens', text='statement', stop_words=None):

    lemmatizer_ = WordNetLemmatizer()

    en = spacy.load("en_core_web_lg")

    if stop_words is None:
        stopwords = en.Defaults.stop_words

    df = df.dropna(subset=[text])

    df[tokens] = df[text].apply(wordopt)

    df[tokens] = df[tokens].apply(lambda x: process_text(x, word_tokenize, stopwords, lemmatizer_))

    return df