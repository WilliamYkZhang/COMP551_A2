# Preprocessing
from sklearn.preprocessing import Normalizer
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import SelectKBest
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

# Module to write final params 
import csv
import datetime
import pickle 

""" Module containing the model selection pipeline """
if __name__  == "__main__":

    # Get a list of stopwords
    stopwords = stopwords.words("english")

    # Transformers 
    c_vect = CountVectorizer(lowercase=True, encoding="utf-8", decode_error="ignore", strip_accents='unicode',stop_words=stopwords, analyzer = "word")
    tfidf_vect = TfidfVectorizer(lowercase=True, encoding = "utf-8",  decode_error = 'ignore', strip_accents='unicode', stop_words=stopwords, analyzer = "word")  
    tfidf_trans = TfidfTransformer()
    svd = TruncatedSVD()
    nml = Normalizer()
    
    # Estimators 
    log_reg = LogisticRegression()
    svc = SVC() # class weight , experiement values 
    xgb = xgb.XGBClassifier()
    decision_tree_clf = DecisionTreeClassifier()
    rff = RandomForestClassifier()
    multi_NB = MultinomialNB()
    
    # Building pipeline 
    pipeline_cvect = Pipeline([('cvect', c_vect), ('clf', multi_NB)], verbose=True)
    pipeline_cvect_svd = Pipeline([('cvect', c_vect),('svd', svd), ("nml", nml), ('clf', multi_NB)], verbose=True)
    pipeline_tfidf = Pipeline([('tfidf', tfidf_vect), ('clf', multi_NB)], verbose=True)
    pipeline_tfidf_svd = Pipeline([('tfidf', tfidf_vect), ('svd', svd), ("nml", nml), ('clf', multi_NB)], verbose=True)
    pipeline_cvect_tfidf = Pipeline([('cvect', c_vect),('tfidf', tfidf_trans), ('kbest', SelectKBest()), ('clf', multi_NB)], verbose=True)
    
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
        'tfidf__max_features': (None, 10000, 25000, 50000),
        'tfidf__use_idf': (True, False), # Enable inverse-document-frequency reweighting.
        'tfidf__max_df': (0.5, 0.75, 0.9), # ignore terms that have a document frequency strictly higher than the given threshold
        'tfidf__min_df': (0.025, 0.05, 0.1), #  ignore terms that have a document frequency strictly lower than the given threshold
        'tfidf__norm': ('l1', 'l2', None), # regularization term
        'tfidf__smooth_idf': (True, False), # Smooth idf weights by adding one to document frequencies, as if an extra document was seen containing every term in the collection exactly once.Prevents zero divisions
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
        'cvect__max_df': (0.5, 0.75, 1.0),
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
    lemmatized_df = pd.read_csv("preprocessed_reddit_train_WordNetLemmatizer.csv")

    # Separate X and Y 
    X_stem = stemmed_df["cleaned"]
    y_stem = stemmed_df["label"]
    X_lemma = lemmatized_df["cleaned"]
    y_lemma = lemmatized_df["label"]
    
    # Use GridSearch cross validation to find the best feature extraction and hyperparameters
    gs_CV = GridSearchCV(pipeline_tfidf, param_grid=parameters_tfidf, cv=5)
    gs_CV.fit(X_lemma, y_lemma)
    print("Performing grid search...")
    print("Pipeline: ", [name for name, _ in pipeline_tfidf.steps])
    print("Best parameter (CV score={0:.3f}):".format(gs_CV.best_score_))
    print("Best parameters set: {} \nBest estimator parameters {}.".format(gs_CV.best_params_, gs_CV.best_estimator_.get_params()))

    # Write best params in csv file 
    with open(r"parameters.csv", "a") as f:
        # To write a csv_file we are using a csv_writer from csv module
        csv_writer = csv.writer(f, delimiter=",", lineterminator="\n")         
        # Write current time
        csv_writer.writerow([datetime.datetime.now()])
        score = "Cross Validation score = " + str(gs_cv.best_score_)
        csv_writer.writerow([score])        
        # Write best parameters
        for key, value in gs_CV.best_params_.items(): 
            csv_writer.writerow([key, value])   
        csv_writer.writerow["Best estimator's parameters"]
        for value in gs_CV.best_estimator_.get_params():
            csv_writer.writerow(value)
 
    pickle.dump(gs_CV.best_estimator_, open("models/best_estimator_.pkl", "wb"))