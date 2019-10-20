from sklearn.ensemble import AdaBoostClassifier
from validation import grid_search_cv, cross_validation
import pandas as pd 

# Read data
df = pd.read_csv("preprocessed_reddit_train_SnowballStemmer.csv")
y_train = df["label"]
X_train = df["cleaned"]

# Create Ada Boosting classifier
clf = AdaBoostClassifier()

# Model parameters
params = {
    'clf__n_estimators': (50, 100,200),
    'clf__learning_rate': (0.5, 1.0,1.5), 
    'clf__algorithm':("SAMME", "SAMME.R"),
}

# TODO: try different base estimators

# Perform Grid Search CV to find the best parameters
print(cross_validation)
