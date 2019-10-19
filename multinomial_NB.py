from sklearn.naive_bayes import MultinomialNB
from predict import classify

multi_NB = MultinomialNB(alpha=0.25)
classify(multi_NB)