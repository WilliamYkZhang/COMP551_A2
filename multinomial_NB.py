from sklearn.naive_bayes import MultinomialNB
from preprocess import reddit_preprocess
from sklearn.metrics import accuracy_score

x_train, x_valid, y_train, y_valid = reddit_preprocess()
classifier = MultinomialNB(alpha=0.5)
classifier.fit(x_train, y_train)
y_pred = classifier.predict(x_valid)
print(accuracy_score(y_pred = y_pred, y_true= y_valid))