""" Class implementing Logistic Regression on the Reddit Dataset """
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from preprocess import reddit_preprocess
# from validation import validate

# Use GridSearchCV to find the best parameters 
if __name__ == "__main__":
    x_train, x_valid, y_train, y_valid = reddit_preprocess()
    
    classifier = LogisticRegression(max_iter==3000)
    classifier.fit(x_train,y_train)
    y_pred = classifier.predict(x_valid)
    print(accuracy_score(y_pred = y_pred, y_true= y_valid))
    # print(validate())
    