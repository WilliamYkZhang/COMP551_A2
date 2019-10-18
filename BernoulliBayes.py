import numpy as np
from sklearn import preprocessing
from scipy.sparse import csc_matrix


class BernoulliBayes:
    _smoothing = 1.
    _nclasses = 0
    _fitParams = None
    _encoder = preprocessing.LabelEncoder()

    def __init__(self, smoothing=1.):
        self._smoothing = smoothing

    def fit(self, trainingSet, trainingLabels):

        self._nclasses = np.amax(trainingLabels) + 1

        # generates list containing a count of each class occurrence
        occurrences = [0] * self._nclasses

        for element in trainingLabels:
            occurrences[element] += 1

        # fit parameter matrix with shape (nclasses, nfeatures + 1)
        params = np.zeros((self._nclasses, trainingSet.shape[1] + 1))

        # fills params with # of feature occurrences per class then divides by # of class occurrences
        for i in range(self._nclasses):
            for n, element in enumerate(trainingLabels):
                if element == i:
                    params[i, :-1] += trainingSet[n]
            params[i, :-1] = (params[i, :-1] + self._smoothing)/(float(occurrences[i]) + 2. * self._smoothing)
            params[i, -1] = occurrences[i]/trainingSet.shape[0]

        self._fitParams = params

    def validate(self, validationSet, validationLabels):

        # creating a log odds matrix
        odds = np.zeros((self._nclasses, validationSet.shape[0]), dtype=np.float32)

        # adding class prior probability
        for Class in range(self._nclasses):
            odds[Class] += np.log(self._fitParams[Class, -1]/(1 - self._fitParams[Class, -1]))

        odds += np.log(self._fitParams[:, :-1]) @ validationSet.T
        odds += (np.log(1 - self._fitParams[:, :-1]).sum(axis=1).reshape((-1, 1))) - (np.log(1 - self._fitParams[:, :-1]) @ validationSet.T)

        predictions = []
        for example in odds.T:
            predictions.append(np.argmax(example))

        print("accuracy: " + str(np.sum(predictions == validationLabels)/len(predictions)))
