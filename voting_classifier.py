# Models
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC, SVC
import pandas as pd 
from validation import cross_validation, grid_search_cv

""" 
Results
clf
0.5550571428571429

clf2
0.5693857142857143

clf3
0.5652999999999999
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
log_reg = LogisticRegression(C=4.5, penalty="l2", multi_class='ovr', solver='liblinear', max_iter=300, dual=False, warm_start=True, fit_intercept=0.4) 
multi_NB = MultinomialNB(alpha=0.225)
linear_svc = LinearSVC(penalty="l2",loss="hinge",  multi_class="ovr", dual=True, fit_intercept=False, intercept_scaling=1.65, max_iter=2000, C=1.1, tol=0.00005)
svc_clf = SVC(probability=True, kernel="linear",  decision_function_shape="ovr",  max_iter=2000, C=1.1, tol=0.00005)

# Voting Classifier
clf = VotingClassifier(estimators=[('lr', log_reg), ("nb", multi_NB), ("svc", svc_clf)], voting="soft")
clf2 = VotingClassifier(estimators=[('lr', log_reg), ("nb", multi_NB), ("svc", linear_svc)], voting="hard")
clf3 = VotingClassifier(estimators=[('lr', log_reg), ("nb", multi_NB)], voting="soft")

# Number of cross validation folds
folds = 5

# Perform cross validation
print(cross_validation(model=clf2, X=X_stem, y=y_stem, folds=folds))

# Perform Grid Search CV 
# print(grid_search_cv(model=log_reg,X=X_stem, y=y_stem,params=params_log_reg, folds=folds))
