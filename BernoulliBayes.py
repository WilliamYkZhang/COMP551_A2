import numpy as np


class BernoulliBayes:
    _smoothing = 1.
    _nclasses = 0
    _fitParams = None
    _fitParams0 = None

    def __init__(self, smoothing=1.):
        self._smoothing = smoothing

    def fit(self, trainingSet):
        self._nclasses = int(np.amax(trainingSet[:, -1])) + 1

        # generates list containing a count of each class occurrence
        occurrences = [0] * self._nclasses
        for i in range(self._nclasses):
            for element in trainingSet[:, -1]:
                if i == element:
                    occurrences[i] = occurrences[i] + 1

        # fit parameter matrix with shape (nclasses, nfeatures + 1)
        params = np.array([[0.] * len(trainingSet[0])] * self._nclasses)

        # fills params with # of feature occurrences per class then divides by # of class occurrences
        for i in range(self._nclasses):
            for row in trainingSet:
                if row[-1] == i:
                    params[i] += np.append(row[:-1], 1/len(trainingSet))
            params[i, :-1] = (params[i, :-1] + self._smoothing)/(float(occurrences[i]) + 2. * self._smoothing)

        self._fitParams = np.array(params)
        print(self._fitParams)

        # fit parameter matrix with shape (nclasses, nfeatures)
        params = np.array([[0.] * (len(trainingSet[0]) - 1)] * self._nclasses)

        # fills params with # of feature occurrences per class then divides by # of class occurrences
        for i in range(self._nclasses):
            for row in trainingSet:
                if row[-1] != i:
                    params[i] += row[:-1]
            params[i] = (params[i] + self._smoothing) / (float((len(trainingSet) - occurrences[i]) + 2. * self._smoothing))

        self._fitParams0 = np.array(params)
        print(self._fitParams0)

    def validate(self, validationSet):
        valParams = self._fitParams
        valParams0 = self._fitParams0

        # if validation set has more features than the training set, fill with smoothed priors
        if len(validationSet[0]) > self._fitParams.shape[1]:
            smoothPrior = self._smoothing/(self._nclasses + 2. * self._smoothing)
            for i in range(len(validationSet[0]) - self._fitParams.shape[1]):
                valParams = np.insert(valParams, -1, [smoothPrior] * self._nclasses, axis=1)

        # if validation set has more features than the training set, fill with smoothed priors
        if len(validationSet[0]) > self._fitParams0.shape[1]:
            smoothPrior = self._smoothing / (self._nclasses + 2. * self._smoothing)
            for i in range(len(validationSet[0]) - self._fitParams0.shape[1]):
                valParams0 = np.insert(valParams0, -1, [smoothPrior] * self._nclasses, axis=1)


            print(valParams)
            print(valParams0)
        predictions = []

        # creating an array of class log odds for each example
        for example in range(len(validationSet)):

            odds = [0.] * self._nclasses

            for Class in range(self._nclasses):
                # adding class prior probability
                odds[Class] = np.log(valParams[Class, -1]/(1 - valParams[Class, -1]))

                # adding log odds per feature
                for feature in range(len(validationSet[:-1])):
                    if validationSet[example, feature] == 1:
                        odds[Class] += np.log(valParams[Class, feature]/(valParams0[Class, feature]))
                    else:
                        odds[Class] += np.log((1 - valParams[Class, feature])/( 1 - (valParams0[Class, feature])))

            #converting log odds to a prediction
            bestOdds = odds[0]
            bestClass = 0
            for i, n in enumerate(odds):
                if n > bestOdds:
                    bestOdds = n
                    bestClass = i

            predictions.append(bestClass)
        print(predictions)






meme = np.array([[0., 0., 0.], [0., 1., 1.], [1., 0., 2.]])

validationMeme = np.array([[0., 0., 1., 1., 0.], [0., 0., 1., 0., 1.], [0., 1., 1., 0., 1.], [1., 1., 0., 1., 0.], [1., 1., 0., 1., 0.]])

test = BernoulliBayes(smoothing=1.)
test.fit(meme)

test.validate(validationMeme)
