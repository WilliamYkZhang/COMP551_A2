from validation import grid_search_cv_svd
from sklearn.naive_bayes import MultinomialNB
import pandas as pd 

""" Module experimenting with n grams """

# Read DataFrame
stemmed_df = pd.read_csv("preprocessed_reddit_train_SnowballStemmer.csv")

# Separate X and Y 
X_stem = stemmed_df["cleaned"]
y_stem = stemmed_df["label"]

parameters_tfidf_svd = {
'tfidf__ngram_range': ((1, 2),(1,3)), # n-grams to be extracted
'svd__n_components' : (10, 500,1000,2500, 5000, 7500, 10000, 20000, 30000,50000),
'svd__algorithm': ("arpack", "randomized"),
'nml__norm' : ('l2', 'max')     
}   

# Instantiate model
multi_NB = MultinomialNB(alpha=0.25)

# Print results
print(grid_search_cv_svd(model= multi_NB, X=X_stem, y=y_stem, params=parameters_tfidf_svd, folds=5))