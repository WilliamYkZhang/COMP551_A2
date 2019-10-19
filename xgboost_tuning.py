
from validation import grid_search_cv, cross_validation
import xgboost as xgb
import pandas as pd 

# Read data
df = pd.read_csv("preprocessed_reddit_train_SnowballStemmer.csv")
y_train = df["label"]
X_train = df["cleaned"]

# Instantiate model
clf = xgb.XGBClassifier(booster='gbtree', eta=0.1)

# Instantiate parameters 
parameters = {
    # 'clf__booster': ('gbtree', 'gblinear'),
    # 'clf__objective':('multi:softmax', 'binary:logistic'),   
    # 'clf__eval_metric':('rmse', 'mae', 'logloss', 'logloss', 'error', 'merror', 'mlogloss', 'auc'),
    # 'clf__seed': (0,1,2,3),
    'clf__eta': (0.01,0.1,0.2), # learning rate
    # 'clf__min_child_weight':(1,3,5,10), # Used to control over-fitting. Higher values prevent a model from learning relations which might be highly specific to the particular sample selected for a tree. Too high values can lead to under-fitting hence, it should be tuned using CV.
    # 'clf__max_depth': (3,6,9),
}   
folds = 2

# Parameter tuning
print(cross_validation(model=clf, X=X_train, y=y_train, folds=folds))
# best_scores, best_params, best_estimator_params = grid_search_cv(model=clf, X=X_train, y=y_train, params=parameters,folds=folds)
# print(type(best_scores))
# print(type(best_params))
# print(type(best_estimator_params))

""" 
Results 
XGBClassifier('gbtree')

[0.42921429 0.43292857 0.43657143 0.43364286 0.434500000] (without parameters, eta=0.3, max_depth=3)
[0.42148571 0.42485714]                                   (eta=0.5, max_depth=6)
(eta=0.1, max_depth=6)
"""