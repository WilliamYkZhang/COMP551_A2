from sklearn.naive_bayes import MultinomialNB
from predict import classify

classifier = MultinomialNB(alpha=0.5)
classify(classifier)