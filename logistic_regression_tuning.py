from sklearn.linear_model import LogisticRegression
import pandas as pd 
from validation import cross_validation, grid_search_cv

""" 
Model: Logistic Regression()
Stemmed 
5 folds: 
0.5377285714285713

Lemmatized
5 folds:
0.5388857142857144
0.5252142857142857 (C=0.5)
0.5429571428571429 (C=1.5)
"""

# Read DataFrame
stemmed_df = pd.read_csv("preprocessed_reddit_train_SnowballStemmer.csv")
lemmatized_df = pd.read_csv("preprocessed_reddit_train_WordNetLemmatizer.csv")

# Separate X and Y 
X_stem = stemmed_df["cleaned"]
y_stem = stemmed_df["label"]
X_lemma = lemmatized_df["cleaned"]
y_lemma = lemmatized_df["label"]

# Model
log_reg = LogisticRegression(C=2)

# Model parameters
params_log_reg = {
    'clf__C': (0.25, 0.5 ,0.75, 1.0, 1.5),
    'clf__penalty': ('l2',),
    'clf__class_weight':('balanced', None),
    'clf__max_iter': (1000, 2500, 5000),
    'clf__multi_class': ('ovr', 'multinomial'), # one vs all or multinomial   
    'clf__solver': ('newton-cg', 'sag', 'lbfgs'),
}

# Number of cross validation folds
folds = 5

print(cross_validation(model=log_reg,X=X_lemma, y=y_lemma, folds=folds))