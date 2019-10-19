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

# Transformers 
c_vect = CountVectorizer(lowercase=True, encoding="utf-8", decode_error="ignore", strip_accents='unicode',stop_words=stopwords.words('english'), analyzer = "word")
tfidf_vect = TfidfVectorizer(smooth_idf=True, norm='l2', lowercase=True, max_features=30000, use_idf=True, encoding = "utf-8",  decode_error = 'ignore', strip_accents='unicode', stop_words=stopwords.words('english'), analyzer = "word")
tfidf_trans = TfidfTransformer()
svd = TruncatedSVD()
nml = Normalizer()

# Pipelines 
# pipeline_cvect = Pipeline([('cvect', c_vect), ('clf', multi_NB)], verbose=True)
# pipeline_cvect_svd = Pipeline([('cvect', c_vect),('svd', svd), ("nml", nml), ('clf', multi_NB)], verbose=True)
# pipeline_tfidf = Pipeline([('tfidf', tfidf_vect), ('clf', multi_NB)], verbose=True)
# pipeline_tfidf_svd = Pipeline([('tfidf', tfidf_vect), ('svd', svd), ("nml", nml), ('clf', multi_NB)], verbose=True)
# pipeline_cvect_tfidf = Pipeline([('cvect', c_vect),('tfidf', tfidf_trans), ('kbest', SelectKBest()), ('clf', multi_NB)], verbose=True)

# Instantiate parameters for pipeline     
parameters_cvect = {
'cvect__max_df': (0.5, 0.75, 1.0),
'cvect__max_features': (None, 5000, 10000, 50000),
'cvect__ngram_range': ((1, 1), (1, 2)),  # unigrams or bigrams
'cvect__max_df': (0.5, 0.75, 0.9), # ignore terms that have a document frequency strictly higher than the given threshold
'cvect__min_df': (0.025, 0.05, 0.1), #  ignore terms that have a document frequency strictly lower than the given threshold
'clf__alpha': (0.25, 0.5, 0.75),
'clf__fit_prior': (True, False),   
}

parameters_cvect_svd = {
'cvect__max_df': (0.5, 0.75, 1.0),
'cvect__max_features': (None, 5000, 10000, 50000),
'cvect__ngram_range': ((1, 1), (1, 2)),  # unigrams or bigrams
'cvect__max_df': (0.5, 0.75, 0.9), # ignore terms that have a document frequency strictly higher than the given threshold
'cvect__min_df': (0.025, 0.05, 0.1), #  ignore terms that have a document frequency strictly lower than the given threshold
'svd__n_components' : (10, 500,1000,2500, 5000, 7500, 10000),
'svd__algorithm': ("arpack", "randomized"),
'nml__norm' : ('l1', 'l2', 'max'),
'clf__alpha': (0.25, 0.5, 0.75),
'clf__fit_prior': (True, False),   
}

parameters_tfidf = {
'tfidf__max_features': (None,1000, 5000, 10000),
'tfidf__use_idf': (True, False), # Enable inverse-document-frequency reweighting.
'tfidf__max_df': (0.75, 0.9), # ignore terms that have a document frequency strictly higher than the given threshold
'tfidf__min_df': (0.05, 0.1), #  ignore terms that have a document frequency strictly lower than the given threshold
'tfidf__norm': ('l1', 'l2', None), # regularization term
'tfidf__smooth_idf': (True, False), # Smooth idf weights by adding one to document frequencies, as if an extra document was seen containing every term in the collection exactly once. Prevents zero divisions
'tfidf__ngram_range': ((1, 1), (1, 2)), # n-grams to be extracted
'clf__alpha': (0.25, 0.5, 0.75),
'clf__fit_prior': (True, False),      
}  

parameters_tfidf_svd = {
'tfidf__max_features': (None, 10000, 25000, 50000),
'tfidf__use_idf': (True, False), # Enable inverse-document-frequency reweighting.
'tfidf__max_df': (0.5, 0.75, 0.9), # ignore terms that have a document frequency strictly higher than the given threshold
'tfidf__min_df': (0.025, 0.05, 0.1), #  ignore terms that have a document frequency strictly lower than the given threshold
'tfidf__norm': ('l1', 'l2', None), # regularization term
'tfidf__smooth_idf': (True, False), # Smooth idf weights by adding one to document frequencies, as if an extra document was seen containing every term in the collection exactly once.Prevents zero divisions
'tfidf__ngram_range': ((1, 1), (1, 2)), # n-grams to be extracted
'svd__n_components' : (10, 500,1000,2500, 5000, 7500, 10000),
'svd__algorithm': ("arpack", "randomized"),
'nml__norm' : ('l1', 'l2', 'max'),
'clf__alpha': (0.25, 0.5, 0.75),
'clf__fit_prior': (True, False),      
}  

parameters_cvect_tfidf = {
'cvect__max_features': (None, 5000, 10000, 50000),
'cvect__ngram_range': ((1, 1), (1, 2)),  # unigrams or bigrams
'cvect__max_df': (0.5, 0.75, 0.9), # ignore terms that have a document frequency strictly higher than the given threshold
'cvect__min_df': (0.025, 0.05, 0.1), #  ignore terms that have a document frequency strictly lower than the given threshold
'tfidf__use_idf': (True, False), # Enable inverse-document-frequency reweighting.
'tfidf__norm': ('l1', 'l2', None), # regularization term
'tfidf__smooth_idf': (True, False), # Smooth idf weights by adding one to document frequencies, as if an extra document was seen containing every term in the collection exactly once.Prevents zero divisions
'tfidf__sublinear': (True, False),
'clf__alpha': (0.25, 0.5, 0.75),
'clf__fit_prior': (True, False),       
}  

# Read DataFrame
stemmed_df = pd.read_csv("preprocessed_reddit_train_SnowballStemmer.csv")
# lemmatized_df = pd.read_csv("preprocessed_reddit_train_WordNetLemmatizer.csv")

# Separate X and Y 
X_stem = stemmed_df["cleaned"]
y_stem = stemmed_df["label"]
# X_lemma = lemmatized_df["cleaned"]
# y_lemma = lemmatized_df["label"]

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
0.5651428571428572 ***BEST


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

# Estimators 
log_reg = LogisticRegression(multi_class='multinomial', n_jobs=-1, solver='newton-cg')
svc = SVC(C = 1.0, kernel = 'rbf') # class weight , experiement values 
xgb = xgb.XGBClassifier(objective='multi:softmax')
decision_tree_clf = DecisionTreeClassifier()
rff = RandomForestClassifier()
multi_NB = MultinomialNB(alpha=0.25)

# Model parameters
params_decision_tree = {
    'clf__alpha': (0.25, 0.5, 0.75),
    'clf__fit_prior': (True, False),    
}

params_rff = {
    'clf__alpha': (0.25, 0.5, 0.75),
    'clf__fit_prior': (True, False),    
}

params_xgboost = {
    'clf__alpha': (0.25, 0.5, 0.75),
    'clf__fit_prior': (True, False),    
}

# print(stopwords.words("english"))
print(cross_validation(model=multi_NB, X=X_stem, y=y_stem, folds=5))
# print(grid_search_cv(model=log_reg, X=X_stem, y=y_stem, params=params_log_reg, folds=5))
# print(grid_search_cv(model=rff, X=X_stem, y=y_stem, params=, folds=5))
# print(grid_search_cv(model=decision_tree_clf, X=X_stem, y=y_stem, params=, folds=5))
