from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

""" Module containing the model selection pipeline """
Class ModelSelector:
    def __init__(self):
        self.pipeline = Pipeline([('vect', CountVectorizer()),('tfidf', TfidfTransformer()),('clf', SGDClassifier(tol=1e-3)),])
        self.parameters = parameters = {
                                        'vect__max_df': (0.5, 0.75, 1.0),
                                        # 'vect__max_features': (None, 5000, 10000, 50000),
                                        'vect__ngram_range': ((1, 1), (1, 2)),  # unigrams or bigrams
                                        # 'tfidf__use_idf': (True, False),
                                        # 'tfidf__norm': ('l1', 'l2'),
                                        'clf__max_iter': (20,),
                                        'clf__alpha': (0.00001, 0.000001),
                                        'clf__penalty': ('l2', 'elasticnet'),
                                        # 'clf__max_iter': (10, 50, 80),
                                        }
def validate(model, features, target):
    return cross_val_score(model, X=features, y=target)
    # we need to transform each fold and train on every n-1 folds. 

def grid_search_cv(model, params):
    return GridSearchCV(model, param_grid = params)