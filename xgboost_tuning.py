
from validation import grid_search_cv, cross_validation
import xgboost as xgb
import pandas as pd 

# Read data
df = pd.read_csv("preprocessed_reddit_train_SnowballStemmer.csv")
y_train = df["label"]
X_train = df["cleaned"]

# Instantiate model
clf = xgb.XGBClassifier(booster='gbtree', learning_rate=0.1)

# Instantiate parameters 
parameters = {
    # 'clf__booster': ('gbtree', 'gblinear'),
    # 'clf__objective':('multi:softmax', 'binary:logistic'),   
    # 'clf__eval_metric':('rmse', 'mae', 'logloss', 'logloss', 'error', 'merror', 'mlogloss', 'auc'),
    # 'clf__seed': (0,1,2,3),
    'clf__eta': (0.0005, 0.001, 0.005), # learning rate
    # 'clf__min_child_weight':(1,3,5,10), # Used to control over-fitting. Higher values prevent a model from learning relations which might be highly specific to the particular sample selected for a tree. Too high values can lead to under-fitting hence, it should be tuned using CV.
    # 'clf__max_depth': (3,6,9),
}   

""" 
Results 
XGBClassifier('gbtree')

"""

# Number of cross validation folds 
folds = 2

# Perform Cross-Validation to validate model 
print(cross_validation(model=clf, X=X_train, y=y_train, folds=folds))

# Perform Grid Search CV to find the best parameters
# best_scores, best_params, best_estimator_params = grid_search_cv(model=clf, X=X_train, y=y_train, params=parameters,folds=folds)
