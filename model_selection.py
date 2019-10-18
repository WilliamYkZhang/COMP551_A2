# Preprocessing
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from nltk.corpus import stopwords
import pandas as pd

# Transformers 
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

# Models 
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC 
import xgboost as xgb
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB

""" Module containing the model selection pipeline """
if __name__  == "__main__":

    # Get a list of stopwords
    stopwords = stopwords.words("english")

    # Transformers 
    c_vect = CountVectorizer(lowercase=True, encoding="utf-8", decode_error="ignore", strip_accents='unicode',stop_words=stopwords, analyzer = "word")
    tfidf_vect = TfidfVectorizer(lowercase=True, encoding = "utf-8",  decode_error = 'ignore', strip_accents='unicode', stop_words=stopwords, analyzer = "word")  
    svd = TruncatedSVD()

    # Estimators 
    log_reg = LogisticRegression()
    svc = SVC() # class weight , experiement values 
    xgb = xgb.XGBClassifier()
    decision_tree_clf = DecisionTreeClassifier()
    rff = RandomForestClassifier()
    multi_NB = MultinomialNB()
    
    # Building pipeline 
    pipeline_cvect = Pipeline([('cvect', c_vect),('svd', svd),('clf', multi_NB),], memory="Pipeline Output/", verbose=True)
    pipeline_tfidf = Pipeline([('tfidf', tfidf_vect), ('clf', multi_NB)], memory="Pipeline Output/", verbose=True)
    pipeline_cvect_tfidf = Pipeline([('cvect', c_vect),('tfidf', tfidf_vect),('svd', svd),('clf', multi_NB),], memory="Pipeline Output/", verbose=True)
    # ('svd', svd),
    # Instantiate parameters for pipeline     
    parameters_cvect = {
        'cvect__max_df': (0.5, 0.75, 1.0),
        'cvect__max_features': (None, 5000, 10000, 50000),
        'cvect__ngram_range': ((1, 1), (1, 2)),  # unigrams or bigrams
        'svd__n_components' : [2,50,100,250, 500,1000,10000,25000],
        'svd__algorithm': ("arpack", "randomized"),
        }

    parameters_tfidf = {
        'tfidf__max_features': (None, 10000, 25000, 50000),
        'tfidf__use_idf': (True, False), # Enable inverse-document-frequency reweighting.
        'tfidf__max_df': (0.5, 0.75, 0.9), # ignore terms that have a document frequency strictly higher than the given threshold
        'tfidf__min_df': (0.025, 0.05, 0.1), #  ignore terms that have a document frequency strictly lower than the given threshold
        'tfidf__norm': ('l1', 'l2', None), # regularization term
        'tfidf__smooth_idf': (True, False), # Smooth idf weights by adding one to document frequencies, as if an extra document was seen containing every term in the collection exactly once.Prevents zero divisions
        'tfidf__ngram_range': ((1, 1), (1, 2)), # n-grams to be extracted
        # 'svd__n_components' : (10, 500,1000,2500, 5000, 7500, 10000),
        # 'svd__algorithm': ("arpack", "randomized"),
        'clf__alpha': (0.25, 0.5, 0.75),
        'clf__fit_prior': (True, False),      
    }  

    parameters_cvect_tfidf = {
        'tfidf__max_features': (None, 5000, 10000, 50000),
        'tfidf__use_idf': (True, False),
        'tfidf__norm': ('l1', 'l2', None),
        'tfidf__ngram_range': ((1, 1), (1, 2)), 
        'svd__n_components' : [2,50,100,250, 500,1000,10000,25000],
        'svd__algorithm': ("arpack", "randomized"),
        'clf__max_iter': (20,),
        'clf__alpha': (0.00001, 0.000001),
        'clf__penalty': ('l2', 'elasticnet'),
        'clf__max_iter': (10, 50, 80),        
    }  

    # Read DataFrame
    stemmed_df = pd.read_csv("preprocessed_reddit_train_SnowballStemmer.csv")
    lemmatized_df = pd.read_csv("preprocessed_reddit_train_WordNetLemmatizer.csv")

    # Separate X and Y 
    X_stem = stemmed_df["cleaned"]
    y_stem = stemmed_df["label"]
    X_lemma = lemmatized_df["cleaned"]
    y_lemma = lemmatized_df["label"]
    
    # Check for number of features.
    gs_CV = GridSearchCV(pipeline_tfidf, param_grid = parameters_tfidf, cv=5)
    gs_CV.fit(X_lemma, y_lemma)
    print("Best parameter (CV score={0:.3f}):".format(gs_CV.best_score_))
    print(gs_CV.best_params_)