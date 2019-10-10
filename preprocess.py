import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords 
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import word_tokenize

""" Module containing preprocessing function for reddit comments """

def reddit_preprocess():
    # Read dataset
    df = pd.read_csv("reddit_train.csv")

    # Apply stemming function 
    df["stemmed"] = df["comments"].apply(stem)
    # print(df["stemmed"])

    # Split feature and targ(et variables 
    X = df["stemmed"] # pandas series
    y = df["subreddits"] # pandas series

    # Split training and validation set
    x_train, x_valid, y_train, y_valid = train_test_split(X, y, train_size = 0.8, test_size = 0.2, shuffle=True)

    vectorized_x_train = vectorize(x_train)
    # Only transform not fit )
    # vectorized_x_test = vectorize(x_valid)

    # print(x_train)
    print(vectorized_x_train)

    # Get rid of &gt
    # print(x_train)
# Remove stop-words stem function 
# PCA 

def vectorize(training_data):
    """ Vectorize text using TfidfVectorizer """
    # Get a list of stopwords
    stop_words = set(stopwords.words('english'))

    # Create a tf_idf word vectorizer 
    tf_idf_vectorizer = TfidfVectorizer(decode_error = 'ignore', strip_accents='unicode', stop_words=stop_words)
    
    return tf_idf_vectorizer.fit_transform(training_data)
    
def stem(sentence):
    stemmed_str = []
    word_tokens = word_tokenize(sentence)
    stemmer = SnowballStemmer("english")
    for i in word_tokens:
        stemmed_str.append(stemmer.stem(i))
    return " ".join(stemmed_str)

if __name__ == "__main__":    
    # from nltk import download

    # download the "all-corpora" corpus
    # download()
    reddit_preprocess()
    
    # Testing different Stemming functions 
    # from nltk.stem.snowball import SnowballStemmer # There are several stemmers that you can use 

    # # Declare language I want to stem in
    # stemmer = SnowballStemmer("english")
    # stemmer.stem("responsiveness")
    # stemmer.stem("responsivity")
    # stemmer.stem("unresponsive") # there are some restriction to this stemmer as it didn't strip the "un"


