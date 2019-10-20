from sklearn.linear_model import LogisticRegression
import pandas as pd 
from validation import cross_validation, grid_search_cv

""" 
Model: Logistic Regression()
Stemmed 
5 folds: 
0.5377285714285713
0.5497714285714286 (C=4.5, penalty="l2")
0.5357142857142858 (C=4.5, penalty="l1")
0.5427714285714285 (solver="sag")
0.5427428571428572 (solver="saga")
0.5427857142857142 (solver="newton-cg")
0.5428857142857143 (solver="lbfgs")
0.5252142857142857 (C=0.5)
0.5429571428571429 (C=1.5)
0.5455857142857142 (C=2)
0.5463285714285714 (C=2.5)
0.5464571428571429 (C=3.0)
0.5469285714285714 (C=3.5)
0.5470428571428572 (C=4.0)
0.5470571428571429 (C=4.5)
0.5466142857142857 (C=5.0)
0.5497714285714286 TUNED

Lemmatized
5 folds:
0.5388857142857144
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

# Model parameters
params_log_reg = {
    'clf__max_iter': (150,250,350),
    'clf__intercept_scaling':(0.3,0.4,0.5)
    # 'clf__multi_class': ('ovr', 'multinomial'), # one vs all or multinomial, hence one vs all is better for logistic regression
    # 'clf__solver': ('newton-cg', 'sag', 'lbfgs','saga'),
}

# Number of cross validation folds
folds = 5

# Perform cross validation
print(cross_validation(model=log_reg, X=X_stem, y=y_stem, folds=folds))

# Perform Grid Search CV 
# print(grid_search_cv(model=log_reg,X=X_stem, y=y_stem,params=params_log_reg, folds=folds))