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

from predict import classify
# Utilities
import datetime
import csv
import pickle
import time

""" Module containing validation pipelines """

def cross_validation(model, X, y, folds):
    pipeline_tfidf = Pipeline([
        ('tfidf', TfidfVectorizer(smooth_idf=True, norm='l2', lowercase=True, max_features=30000,
                                  use_idf=True, encoding = "utf-8",  decode_error = 'ignore',
                                  strip_accents='unicode',
                                  stop_words=stopwords.words('english').append(["nt", "get", "like", "would","peopl", "one", "think", "time", "becaus"]), analyzer="word",
                                  sublinear_tf=True, )),
        ('clf', model)],
         verbose=True)
    # Track CV time
    start = time.time()
    return "Cross validation scores: {0}\nValidation time: {1}".format(cross_val_score(pipeline_tfidf, X, y, cv=folds).mean(),time.time()-start)

def grid_search_cv(model, X, y, params, folds):
    # Pipeline
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(smooth_idf=True, norm='l2', lowercase=True, max_features=30000, use_idf=True, encoding = "utf-8",  decode_error = 'ignore', strip_accents='unicode', stop_words=stopwords.words('english'), analyzer = "word")),
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
        csv_writer.writerow["Best estimator's parameters"]
        for value in gs_CV.best_estimator_.get_params():
            csv_writer.writerow(value)
 
    pickle.dump(gs_CV.best_estimator_, open("models/best_estimator_{}.pkl".format(type(model).__name__), "wb"))

    return (gs_CV.best_score_,gs_CV.best_params_, gs_CV.best_estimator_.get_params())

def grid_search_cv_svd(model, X, y, params, folds):
    # Pipeline
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(smooth_idf=True, norm='l2', lowercase=True, max_features=30000, use_idf=True, encoding = "utf-8",  decode_error = 'ignore', strip_accents='unicode', stop_words=stopwords.words('english'), analyzer = "word")),
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
        csv_writer.writerow["Best estimator's parameters"]
        for value in gs_CV.best_estimator_.get_params():
            csv_writer.writerow(value)
 
    pickle.dump(gs_CV.best_estimator_, open("models/best_estimator_{}.pkl".format(type(model).__name__), "wb"))

    return (gs_CV.best_score_,gs_CV.best_params_, gs_CV.best_estimator_.get_params())

if __name__ == "__main__":
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.ensemble import VotingClassifier
    from sklearn.linear_model import LogisticRegression
    import pandas as pd

    decision_tree_clf = DecisionTreeClassifier()
    rff = RandomForestClassifier(bootstrap=True, max_depth=None,
                                 n_estimators=250, min_samples_leaf=5,
                                 min_samples_split=2)


    nb = MultinomialNB(alpha=0.25)

    clf2 = LogisticRegression(penalty='l2', C=2, max_iter=1000,
                              multi_class='auto', solver='liblinear')
    # Read DataFrame
    stemmed_df = pd.read_csv("preprocessed_reddit_train_SnowballStemmer.csv")
    lemmatized_df = pd.read_csv("preprocessed_reddit_train_WordNetLemmatizer.csv")

    # Separate X and Y
    X_stem = stemmed_df["cleaned"]
    y_stem = stemmed_df["label"]
    X_lemma = lemmatized_df["cleaned"]
    y_lemma = lemmatized_df["label"]

    parameters_tfidf = {
        'clf__bootstrap': [True],  # default
        'clf__max_depth': [100, 1000, None],  # default is none
        'clf__max_features': ['auto'],  # default
        'clf__min_samples_leaf': [0.3, 1, 8],  # default is 1
        'clf__min_samples_split': [0.2, 0.5, 2],  # default is 2
        'clf__n_estimators': [250, 1000]
        # default is 10, but when set to 100 the accuracy increased by 5%, try 1000 tomorrow morning after this training
    }

    # grid_search_cv(model=rff, X=X_stem, y=y_stem, params=parameters_tfidf, folds=5)

    eclf1 = VotingClassifier(estimators=[('nb', nb), ('lr', clf2)], voting='soft')

    # print(cross_validation(model=nb, X=X_stem, y=y_stem, folds=5))

    classify(eclf1)

