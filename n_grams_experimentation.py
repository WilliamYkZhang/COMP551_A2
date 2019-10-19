""" Module experimenting with n grams """

parameters_tfidf_svd = {
'tfidf__max_features': (None, 10000, 25000, 50000),
'tfidf__use_idf': (True, False), # Enable inverse-document-frequency reweighting.
'tfidf__max_df': (0.5, 0.75, 0.9), # ignore terms that have a document frequency strictly higher than the given threshold
'tfidf__min_df': (0.025, 0.05, 0.1), #  ignore terms that have a document frequency strictly lower than the given threshold
'tfidf__norm': ('l1', 'l2', None), # regularization term
'tfidf__smooth_idf': (True, False), # Smooth idf weights by adding one to document frequencies, as if an extra document was seen containing every term in the collection exactly once.Prevents zero divisions
'tfidf__ngram_range': ((1, 1), (1, 2)), # n-grams to be extracted
'svd__n_components' : (10, 500,1000,2500, 5000, 7500, 10000),
'svd__algorithm': ("arpack", "randomized"),
'nml__norm' : ('l1', 'l2', 'max'),
'clf__alpha': (0.25, 0.5, 0.75),
'clf__fit_prior': (True, False),      
}  