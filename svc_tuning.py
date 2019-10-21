from validation import grid_search_cv, cross_validation
from sklearn.svm import SVC
import pandas as pd 

# Read data
df = pd.read_csv("preprocessed_reddit_train_SnowballStemmer.csv")
y_train = df["label"]
X_train = df["cleaned"]

# Instantiate model
clf = SVC(probability=True, kernel="linear",  decision_function_shape="ovr",  max_iter=2000, C=1.1, tol=0.00005)

""" 
Results 
SVC(probability=True, kernel="linear",  decision_function_shape="ovr", max_iter=2000, C=1.1, tol=0.00005)

"""

# Number of cross validation folds 
folds = 5

# Perform Cross-Validation to validate model 
print(cross_validation(model=clf, X=X_train, y=y_train, folds=folds))

# Perform Grid Search CV to find the best parameters
# best_scores, best_params, best_estimator_params = grid_search_cv(model=clf, X=X_train, y=y_train, params=parameters,folds=folds)
