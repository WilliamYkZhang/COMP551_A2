from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from validation import grid_search_cv, cross_validation
import pandas as pd 

# Read data
df = pd.read_csv("preprocessed_reddit_train_SnowballStemmer.csv")
y_train = df["label"]
X_train = df["cleaned"]

# Create Ada Boosting classifier
clf = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=4), n_estimators=1000, algorithm="SAMME.R", learning_rate=0.1)

# Model parameters
params = {
    'clf__n_estimators': (50, 100,200),
    'clf__learning_rate': (0.5, 1.0,1.5), 
    'clf__algorithm':("SAMME", "SAMME.R"),
}

# Number of cross validation folds
folds = 2 

""" 
Results 
AdaBoost(base_estimator=DecisionTreeClassifier(max_depth=4), n_estimators=1000, algorithm="SAMME.R", learning_rate=0.1)

"""

# Perform Cross-Validation to validate model 
print(cross_validation(model=clf, X=X_train, y=y_train, folds=folds))

# Perform Grid Search CV to find the best parameters
# best_scores, best_params, best_estimator_params = grid_search_cv(model=clf, X=X_train, y=y_train, params=params, folds=folds)