from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

""" Module containing the model selection pipeline """

def validate(model, features, target):
    return cross_val_score(model, X=features, y=target)
    # we need to transform each fold and train on every n-1 folds. 

def grid_search_cv(model, params):
    return GridSearchCV(model, param_grid = params)