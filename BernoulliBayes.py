import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import word_tokenize


class BernoulliBayes:
    _smoothing = 1.
    _nclasses = 0
    _fitParams = None
    _encoder = preprocessing.LabelEncoder()

    def __init__(self, smoothing=1.):
        self._smoothing = smoothing

    def fit(self, trainingSet, trainingLabels):

        self._nclasses = np.amax(trainingLabels) + 1

        # generates list containing a count of each class occurrence
        occurrences = [0] * self._nclasses

        for element in trainingLabels:
            occurrences[element] += 1

        # fit parameter matrix with shape (nclasses, nfeatures + 1)
        params = np.zeros((self._nclasses, trainingSet.shape[1] + 1))

        # fills params with # of feature occurrences per class then divides by # of class occurrences
        for i in range(self._nclasses):
            for n, element in enumerate(trainingLabels):
                if element == i:
                    params[i, :-1] += trainingSet[n]
            params[i, :-1] = (params[i, :-1] + self._smoothing)/(float(occurrences[i]) + 2. * self._smoothing)
            params[i, -1] = occurrences[i]/trainingSet.shape[0]

        self._fitParams = params

    def validate(self, validationSet, validationLabels):

        # creating a log odds matrix
        odds = np.zeros((self._nclasses, validationSet.shape[0]), dtype=np.float32)

        # adding class prior probability
        for Class in range(self._nclasses):
            odds[Class] += np.log(self._fitParams[Class, -1]/(1 - self._fitParams[Class, -1]))

        odds += np.log(self._fitParams[:, :-1]) @ validationSet.T
        odds += (np.log(1 - self._fitParams[:, :-1]).sum(axis=1).reshape((-1, 1))) - (np.log(1 - self._fitParams[:, :-1]) @ validationSet.T)

        predictions = []
        for example in odds.T:
            predictions.append(np.argmax(example))

        print("accuracy: " + str(np.sum(predictions == validationLabels)/len(predictions)))


def preprocess():
    # Read dataset
    df = pd.read_csv("reddit_train.csv")

    # Apply stemming function
    df["stemmed"] = df["comments"].apply(stem)

    # Transform each subreddit into an unique integer
    labels, levels = pd.factorize(df["subreddits"])
    df["labels"] = pd.Series(labels)

    # # Split feature and targ(et variables
    X = df["stemmed"]  # pandas series
    y = df["labels"]  # pandas series

    # Split training and validation set
    x_train, x_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2, shuffle=True)

    # Vectorize training data

    vectorized_x_train, vectorized_x_valid = bernoulli_vectorize(x_train, x_valid)
    # Only transform not fit
    return vectorized_x_train, y_train, vectorized_x_valid, y_valid

def stem(sentence):
    stemmed_str = []
    word_tokens = word_tokenize(sentence)
    stemmer = SnowballStemmer("english")
    for i in word_tokens:
        stemmed_str.append(stemmer.stem(i))
    return " ".join(stemmed_str)

def bernoulli_vectorize(training_data, validation_data):
    """ Vectorize text using CountVectorizer """
    # Get a list of stopwords
    stop_words = set(stopwords.words('english'))

    # Create a hashing word vectorizer
    count_vectorizer = CountVectorizer(decode_error='ignore', strip_accents='unicode', stop_words=stop_words, analyzer="word", binary=True)

    return count_vectorizer.fit_transform(training_data), count_vectorizer.transform(validation_data)

if __name__ == "__main__":
    test = BernoulliBayes(0.5)
    testSet, testLabels, validSet, validLabels = preprocess()
    test.fit(testSet, testLabels)
    test.validate(validSet, validLabels)