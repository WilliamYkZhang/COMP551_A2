import pandas as pd
from preprocess import Preprocessor

""" Module that contains a function to predict """

def classify(model):
    """ Function that fits a model using the entire training set and stores its predictions on the held out test set in a csv file. """
   
    # Read datasets
    df = pd.read_csv("preprocessed_reddit_train_SnowballStemmer.csv")

    # Using preprocessor to transform data into tf-idf representation
    preprocessor = Preprocessor("stemmer")

    # Transform training data to tf_idf representation
    x_train = preprocessor.tf_idf_vectorizer.fit_transform(df["cleaned"])
    y_train = df["label"]
    
    # Preprocess test data and transform to tf_idf representation
    x_test_df = pd.read_csv("preprocessed_reddit_test_SnowballStemmer.csv")
    x_test = preprocessor.tf_idf_vectorizer.transform(x_test_df["cleaned"].values.astype('U'))

    # Train model using whole training set
    model.fit(x_train, y_train)

    # Predict on test set
    predictions = model.predict(x_test)

    # Turn predictions back to original labels
    preprocessor.label_encoder.fit(df["subreddits"])
    predictions = preprocessor.label_encoder.inverse_transform(predictions)
    
    # save predictions 
    pred_df =pd.DataFrame({"Id": x_test_df.id , "Category": predictions})
    pred_df.to_csv("predictions.csv", index=False)