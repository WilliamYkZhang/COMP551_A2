# Preprocessing
from sklearn.preprocessing import Normalizer
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import SelectKBest
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from nltk.corpus import stopwords
import pandas as pd

# Transformers 
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer 
from sklearn.decomposition import TruncatedSVD

# Models 
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC 
import xgboost as xgb
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB

# Utilities 
import csv
import datetime
import pickle 

# Import validation functions 
from validation import cross_validation, grid_search_cv

""" 
Results
All scores are mean cross-validation scores 
Model: Multinomial NB()
Stemmed
5 folds: 
0.3420285714285714 (max_features = 1000) 
0.5112142857142857 (max_features = 5000)
0.5448857142857143 (max_features = 10000)
0.5569000000000000 (max_features = 20000)
0.5577142857142856 (max_features = 25000)
0.5573857142857143 (max_features = 30000)
0.5347428571428571 (ngram_range=(1,2), norm='l2')
0.5370000000000000 (ngram_range=(1,2), norm='l1')
0.5348714285714286 (ngram_range=(1,3), norm='l1')
0.5573857142857143 (max_features = 30000, use_idf = True)
0.5573857142857143 (max_features = 30000, use_idf = True, norm='l2')
0.5372857142857143 (max_features = 30000, use_idf = True, norm='l1')
0.5573857142857143 (max_features = 30000, use_idf = True, max_df=0.6)
0.5573857142857143 (max_features = 30000, use_idf = True, max_df=0.9)
0.2954428571428572 (max_features = 30000, use_idf = True, min_df=0.05)
0.5469857142857142 (max_features = 30000, use_idf = True, min_df=0.001)
0.5250428571428571 (max_features = 30000, use_idf = False)
0.5573714285714286 (max_features = 35000)
0.5571428571428572 (max_features = 40000)
0.5571428571428572 (max_features = 50000)

Tuning Multinomial Naive Bayes hyperparameters on 5 folds: 
0.5573857142857143 (alpha = 1)
0.5640857142857143 (alpha = 0.15)
0.5643857142857143 (alpha = 0.20)
0.5647000000000000 (alpha = 0.25)
0.5640000000000001 (alpha = 0.30)
0.5634285714285714 (alpha = 0.35)
0.5624142857142858 (alpha = 0.5)
0.5602428571428572 (alpha = 0.75)

Added stopwords 
0.5675142857142857 BEST


10 folds: 0.5609857142857144
100 folds: 0.5643571428571428
100 folds: 0.5717571428571429 TUNED
1000 folds: 0.5639875000000001


Lemmatized
3 folds: 0.5482855547765574
5 folds: 0.5561857142857143
5 folds: 0.5527142857142857 (max_features = 20000)
5 folds: 0.5561142857142857 (max_features = 30000)
5 folds: 0.5560714285714285 (max_features = 40000)
10 folds: 0.5603857142857143
100 folds: 0.5641714285714287

Model: XGBoost 
Stemmed
5 folds: 

Model: SVC 
Stemmed 

Lemmatized 
"""


# Read DataFrame
stemmed_df = pd.read_csv("preprocessed_reddit_train_SnowballStemmer.csv")
# lemmatized_df = pd.read_csv("preprocessed_reddit_train_WordNetLemmatizer.csv")

# Separate X and Y 
X_stem = stemmed_df["cleaned"]
y_stem = stemmed_df["label"]
# X_lemma = lemmatized_df["cleaned"]
# y_lemma = lemmatized_df["label"]

# Estimators 
multi_NB = MultinomialNB(alpha=0.225)

# Model parameters
params = {
    'clf__alpha':(0.225, 0.25, 0.275),
}

# Number of folds for Cross Validation


# Perform Cross-Validation to validate model 
print(cross_validation(model=multi_NB, X=X_stem, y=y_stem, folds=folds))

# Perform Grid Search CV to find the best parameters
# best_scores, best_params, best_estimator_params = grid_search_cv(model=multi_NB, X=X_stem, y=y_stem, params=params, folds=5)
