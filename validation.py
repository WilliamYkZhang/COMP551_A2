# Preprocessing
from sklearn.preprocessing import Normalizer
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import SelectKBest
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from nltk.corpus import stopwords

# Transformers 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

# Utilities
import datetime
import csv
import pickle
import time

""" Module containing validation pipelines """

def cross_validation(model, X, y, folds):
    pipeline_tfidf = Pipeline([
        ('tfidf', TfidfVectorizer(sublinear_tf=True, stop_words=stopwords.words('english').append(["nt", "get", "like", "would","peopl", "one", "think", "time", "becaus"]), smooth_idf=True, norm="l2",lowercase=True, max_features=30000, use_idf=True, encoding = "utf-8",  decode_error = 'ignore', strip_accents='unicode',  analyzer = "word")),
        ('clf', model)],
         verbose=True)
    # Track CV time
    start = time.time()

    # Scores
    scores = cross_val_score(pipeline_tfidf, X, y, cv=folds)

    return "Cross validation scores: {0}\nCross validation mean score: {1}\nValidation time: {2}s".format(scores, scores.mean(),time.time()-start) 

def grid_search_cv(model, X, y, params, folds):
    # Pipeline
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(sublinear_tf=True, stop_words=stopwords.words('english').append(["nt", "get", "like", "would","peopl", "one", "think", "time", "becaus"]), smooth_idf=True, norm="l2",lowercase=True, max_features=30000, use_idf=True, encoding = "utf-8",  decode_error = 'ignore', strip_accents='unicode',  analyzer = "word")),
        ('clf', model)],
         verbose=True)

    # Use GridSearch cross validation to find the best feature extraction and hyperparameters
    gs_CV = GridSearchCV(pipeline, param_grid=params, cv=folds)
    gs_CV.fit(X, y)
    print("Performing grid search...")
    print("Pipeline: ", [name for name, _ in pipeline.steps])
    print("Best parameter (CV score={0:.3f}):".format(gs_CV.best_score_))
    print("Best parameters set: {} \nBest estimator parameters {}.".format(gs_CV.best_params_, gs_CV.best_estimator_.get_params()))

    # Write best params in csv file 
    with open(r"parameters.csv", "a") as f:
        # To write a csv_file we are using a csv_writer from csv module
        csv_writer = csv.writer(f, delimiter=",", lineterminator="\n")         
        # Write current time
        csv_writer.writerow([datetime.datetime.now()])
        score = "Cross Validation score = " + str(gs_CV.best_score_)
        csv_writer.writerow([score])        
        # Write best parameters
        for key, value in gs_CV.best_params_.items(): 
            csv_writer.writerow([key, value])   
 
    pickle.dump(gs_CV.best_estimator_, open("models/best_estimator_{}.pkl".format(type(model).__name__), "wb"))

    return (gs_CV.best_score_,gs_CV.best_params_, gs_CV.best_estimator_.get_params())

def grid_search_cv_svd(model, X, y, params, folds):
    # Pipeline
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(sublinear_tf=True, stop_words=stopwords.words('english').append(["nt", "get", "like", "would","peopl", "one", "think", "time", "becaus"]), smooth_idf=True, norm="l2",lowercase=True, max_features=30000, use_idf=True, encoding = "utf-8",  decode_error = 'ignore', strip_accents='unicode',  analyzer = "word")),
        ('svd', TruncatedSVD()),
        ('nml', Normalizer()),
        ('clf', model)],
         verbose=True)

    # Use GridSearch cross validation to find the best feature extraction and hyperparameters
    gs_CV = GridSearchCV(pipeline, param_grid=params, cv=folds)
    gs_CV.fit(X, y)
    print("Performing grid search...")
    print("Pipeline: ", [name for name, _ in pipeline.steps])
    print("Best parameter (CV score={0:.3f}):".format(gs_CV.best_score_))
    print("Best parameters set: {} \nBest estimator parameters {}.".format(gs_CV.best_params_, gs_CV.best_estimator_.get_params()))

    # Write best params in csv file 
    with open(r"parameters.csv", "a") as f:
        # To write a csv_file we are using a csv_writer from csv module
        csv_writer = csv.writer(f, delimiter=",", lineterminator="\n")         
        # Write current time
        csv_writer.writerow([datetime.datetime.now()])
        score = "Cross Validation score = " + str(gs_CV.best_score_)
        csv_writer.writerow([score])        
        # Write best parameters
        for key, value in gs_CV.best_params_.items(): 
            csv_writer.writerow([key, value])   
 
    pickle.dump(gs_CV.best_estimator_, open("models/best_estimator_{}.pkl".format(type(model).__name__), "wb"))

    return (gs_CV.best_score_,gs_CV.best_params_, gs_CV.best_estimator_.get_params())