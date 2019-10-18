import numpy as np
import pandas as pd
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
from keras.callbacks import EarlyStopping
from keras.layers import Dropout
import matplotlib.pyplot as plt
import pickle



def preprocess(MAX_NB_WORDS = 50000, MAX_SEQUENCE_LENGTH = 500):

    #MAX_NB_WORDS is equal to the maximum amounts of words to be used in the vector space
    #MAX_SEQUENCE_LENGTH is the maximum amount of words for each comment
    #EMBEDDING_DIM is fixed? why is this look it up.

    df = pd.read_csv('reddit_train.csv')

    # print(df.head(5))
    # print(df.info())
    # print(df['subreddits'].unique())

    tokenizer = Tokenizer(num_words=MAX_NB_WORDS, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True)
    tokenizer.fit_on_texts(df['comments'].values)
    word_index = tokenizer.word_index
    print('Found %s unique tokens.' % len(word_index))

    X = tokenizer.texts_to_sequences(df['comments'].values)
    X = pad_sequences(X, maxlen=MAX_SEQUENCE_LENGTH)
    print('Shape of data tensor:', X.shape)


    Y = pd.get_dummies(df['subreddits']).values
    print('Shape of label tensor:', Y.shape)

    return X, Y

def preprocess_test(MAX_NB_WORDS = 100000, MAX_SEQUENCE_LENGTH = 500):
    df = pd.read_csv('reddit_test.csv')

    tokenizer = Tokenizer(num_words=MAX_NB_WORDS, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True)
    tokenizer.fit_on_texts(df['comments'].values)
    word_index = tokenizer.word_index
    print('Found %s unique tokens.' % len(word_index))

    X = tokenizer.texts_to_sequences(df['comments'].values)
    X = pad_sequences(X, maxlen=MAX_SEQUENCE_LENGTH)
    print('Shape of data tensor:', X.shape)

    return X


def train(X, Y, MAX_NB_WORDS = 100000, EMBEDDING_DIM = 100):

    X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.10, random_state = 42)
    print(X_train.shape, Y_train.shape)
    print(X_test.shape, Y_test.shape)


    model = Sequential()
    model.add(Embedding(MAX_NB_WORDS, EMBEDDING_DIM, input_length=X.shape[1]))
    model.add(SpatialDropout1D(0.2))
    model.add(LSTM(250, dropout=0.2, recurrent_dropout=0.2))
    model.add(LSTM(250, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(20, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    epochs = 5
    batch_size = 64

    history = model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size, validation_split=0.1,
                        callbacks=[EarlyStopping(monitor='val_loss', patience=3, min_delta=0.0001)])



    model.save('2_LSTM_250.h5')
    model.save_weights('2_LSTM_250_weights.h5')

    accr = model.evaluate(X_test,Y_test)
    print('Test set\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}'.format(accr[0],accr[1]))


    plt.title('Loss')
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='test')
    plt.legend()
    plt.show()


    plt.title('Accuracy')
    plt.plot(history.history['acc'], label='train')
    plt.plot(history.history['val_acc'], label='test')
    plt.legend()
    plt.show()


    return model

def predict_test(model):

    X_test = preprocess_test()

    y_predict = model.predict(X_test)

    # y_predict


