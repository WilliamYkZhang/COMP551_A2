# Common Data Science Library
import pandas as pd
import numpy as np

# Sklearn
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import normalize 

# nltk library
from nltk.corpus import stopwords 
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Common python libraries
import re 
import string

# 3rd party package
import unidecode 

""" Module containing preprocessing function for reddit comments """
def reddit_preprocess():
    # Read dataset
    df = pd.read_csv("reddit_train.csv")

    # Apply stemming function 
    df["cleaned"] = df["comments"].apply(clean_text)
    # print(df["comments"])
    print(df["cleaned"])

    # Transform each subreddit into an unique integer
    labels, levels = pd.factorize(df["subreddits"])
    df["labels"] = pd.Series(labels)

    # Split feature and target variables 
    X = df["cleaned"] # pandas series
    y = df["labels"] # pandas series

    # Split training and validation set
    x_train, x_valid, y_train, y_valid = train_test_split(X, y, train_size = 0.8, test_size = 0.2, shuffle=True)

    # Get a list of stopwords
    stop_words = set(stopwords.words('english'))

    # Create a tf_idf word vectorizer 
    tf_idf_vectorizer = TfidfVectorizer(lowercase=True,  decode_error = 'ignore', strip_accents='unicode', stop_words=stop_words, analyzer = "word")  
    # ngram_range = (1,2),

    # Vectorize data 
    x_train = tf_idf_vectorizer.fit_transform(x_train)
    x_valid = tf_idf_vectorizer.transform(x_valid)

    # TODO: Implement Regularization (i.e. PCA) 
    # TODO: Append comment ID to the prediction

    return x_train, x_valid, y_train, y_valid

def clean_text(sentence): 
    # TODO: Remove links

    # Put all words to lower case 
    sentence = sentence.lower()    
    # Tokenize words   
    word_tokens = word_tokenize(sentence)
    # Remove punctuation 
    word_tokens = [_ for _ in word_tokens if _ not in string.punctuation]
    # Strip accents
    word_tokens = [unidecode.unidecode(_) for _ in word_tokens]
    # Remove non-alphabetical char 
    word_tokens = [re.sub(pattern="[^a-zA-Z0-9\s]", repl= "",string=_) for _ in word_tokens]
    # Remove empty strings 
    word_tokens = [_ for _ in word_tokens if _]
    # Stem words
    processed_sentence = stem(" ".join(word_tokens))

    return processed_sentence

def stem(sentence):
    stemmed_str = []
    word_tokens = word_tokenize(sentence)
    stemmer = SnowballStemmer("english")
    for i in word_tokens:
        stemmed_str.append(stemmer.stem(i))
    return " ".join(stemmed_str)

def preprocess_test():
    # Read dataset
    df = pd.read_csv("reddit_train.csv")
    test = pd.read_csv("reddit_test.csv")

    # Apply stemming function 
    df["cleaned"] = df["comments"].apply(clean_text)
    test["cleaned"] = test["comments"].apply(clean_text)

    # Transform each subreddit into an unique integer
    labels, levels = pd.factorize(df["subreddits"])
    df["labels"] = pd.Series(labels)

    # Split feature and target variables 
    X = df["cleaned"] # pandas series
    y = df["labels"] # pandas series

    # Get a list of stopwords
    stop_words = set(stopwords.words('english'))

    # Create a tf_idf word vectorizer 
    tf_idf_vectorizer = TfidfVectorizer(lowercase=True,  decode_error = 'ignore', strip_accents='unicode', stop_words=stop_words, analyzer = "word")  

    # Vectorize data 
    x_train = tf_idf_vectorizer.fit_transform(X)
    x_test = tf_idf_vectorizer.transform(test["cleaned"])

    print(test["cleaned"])
    # TODO: Implement Regularization (i.e. PCA) 
    # TODO: Append comment ID to the prediction

    return x_train, x_test, y


if __name__ == "__main__":    
    # download the "all-corpora" corpus
    # download() # only need to be executed once in order to download language packages from nltk

    # from nltk import download
    # clean_text("L'été dernier, je n'ai pas joué au soccer.")
    # clean_text("I want to si if i'm i'm i'm wasn't didn't")
    # clean_text("I'm testing my unidecode function !@#@$#%$%^%$&^^")
    # print(clean_text("different', 'wording', '**please', 'use', 'one', 'of', 'these', 'subreddits', '**', 'need', 'advice', '/r/advice', 'or', '/r/needadvice', 'or', '/r/relationship_advice', 'cant', 'remember', 'something', '/r/tipofmytongue', 'looking', 'for', 'a', 'particular', 'subreddit', '/r/findareddit', 'otherwise, if these don't fit your needs check out the multireddits"))

    x_train, x_valid, y_train, y_valid = reddit_preprocess()
    print(x_train.shape)
    # print("X train", x_train)
    # print("X valid" , x_valid)
    # print("Y train" , y_train)
    # print("Y valid" , y_valid)


    