from sklearn.naive_bayes import MultinomialNB
from preprocess import reddit_preprocess, preprocess_test
from sklearn.metrics import accuracy_score
import pickle

# X, x_test, y = preprocess_test()
# # x_train, x_valid, y_train, y_valid = reddit_preprocess()
# classifier = MultinomialNB(alpha=0.5)
# classifier.fit(X, y)
# y_pred = classifier.predict(x_test)
# pickle.dump(y_pred, open("predictions.pkl", "wb"))
pred = pickle.load(open("predictions.pkl", "rb"))
print(type(pred))

import pandas as pd

df = pd.read_csv("reddit_test.csv")
df["predictions"] = pred 
print(df)
# print(accuracy_score(y_pred = y_pred, y_true= y_valid))

import numpy as np

df = pd.read_csv('reddit_train.csv')
subreddits = np.asarray(sorted(df['subreddits'].unique()))
print(subreddits)   
labels = (subreddits[np.argmax(pred, axis=1)])
print(labels)