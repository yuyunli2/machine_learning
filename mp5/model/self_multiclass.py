import numpy as np
from sklearn import svm
from sklearn.svm import LinearSVC


class MulticlassSVM:

    def __init__(self, mode):
        if mode != 'ovr' and mode != 'ovo' and mode != 'crammer-singer':
            raise ValueError('mode must be ovr or ovo or crammer-singer')
        self.mode = mode

    def fit(self, X, y):
        if self.mode == 'ovr':
            self.fit_ovr(X, y)
        elif self.mode == 'ovo':
            self.fit_ovo(X, y)
        elif self.mode == 'crammer-singer':
            self.fit_cs(X, y)

    def fit_ovr(self, X, y):
        self.labels = np.unique(y)
        self.binary_svm = self.bsvm_ovr_student(X, y)

    def fit_ovo(self, X, y):
        self.labels = np.unique(y)
        self.binary_svm = self.bsvm_ovo_student(X, y)

    def fit_cs(self, X, y):
        self.labels = np.unique(y)
        X_intercept = np.hstack([X, np.ones((len(X), 1))])

        N, d = X_intercept.shape
        K = len(self.labels)

        W = np.zeros((K, d))

        n_iter = 1500
        learning_rate = 1e-8
        for i in range(n_iter):
            W -= learning_rate * self.grad_student(W, X_intercept, y)

        self.W = W

    def predict(self, X):
        if self.mode == 'ovr':
            return self.predict_ovr(X)
        elif self.mode == 'ovo':
            return self.predict_ovo(X)
        else:
            return self.predict_cs(X)

    def predict_ovr(self, X):
        scores = self.scores_ovr_student(X)
        return self.labels[np.argmax(scores, axis=1)]

    def predict_ovo(self, X):
        scores = self.scores_ovo_student(X)
        return self.labels[np.argmax(scores, axis=1)]

    def predict_cs(self, X):
        X_intercept = np.hstack([X, np.ones((len(X), 1))])
        return np.argmax(self.W.dot(X_intercept.T), axis=0)

    def bsvm_ovr_student(self, X, y):
        '''
        Train OVR binary classfiers.

        Arguments:
            X, y: training features and labels.

        Returns:
            binary_svm: a dictionary with labels as keys,
                        and binary SVM models as values.
        '''
        # print('X', X.shape)
        # print('X type', type(X[1]))
        # print('y', y.shape)
        # print('y type', type(y[1]))
        # data = {}
        #
        # # classify label
        # for i in range(len(y)):
        #     if y[i] in data:
        #         data[y[i]].append(X[i])
        #     else:
        #         data[y[i]] = []
        #         data[y[i]].append(X[i])
        #
        #
        #
        # for ele in data:
        #     data[ele] = np.array(data[ele])
        #
        # print('data in ovr', data[0].shape)
        #
        # binary_svm = {}
        #
        # for ele in data:
        #     binary_svm[ele] = LinearSVC(random_state=12345)
        #
        # for ele in data:
        #
        #     xTrain = data[ele]
        #     yTrain = np.ones((data[ele].shape[0], ))
        #     for num in data:
        #         if ele != num:
        #             xTrain = np.append(xTrain, data[num], axis=0)
        #             yTrain = np.append(yTrain, np.zeros((data[num].shape[0], )))
        #
        #     # print('xTrain', xTrain)
        #     # print('yTrain', yTrain)
        #
        #     binary_svm[ele].fit(xTrain, yTrain)

        binary_svm = {}

        yLabel = np.unique(y)
        for i in yLabel:
            copyY = np.copy(y)
            model = LinearSVC(random_state=12345)

            for j in range(len(copyY)):
                if copyY[j] == yLabel[i]:
                    copyY[j] = 1
                else:
                    copyY[j] = 0

            model.fit(X, copyY)
            binary_svm[yLabel[i]] = model

        return binary_svm

    def bsvm_ovo_student(self, X, y):
        '''
        Train OVO binary classfiers.

        Arguments:
            X, y: training features and labels.

        Returns:
            binary_svm: a dictionary with label pairs as keys,
                        and binary SVM models as values.
        '''
        # print('ovo')
        data = {}

        # classify label
        for i in range(len(y)):
            if y[i] in data:
                data[y[i]].append(X[i])
            else:
                data[y[i]] = []
                data[y[i]].append(X[i])

        for ele in data:
            data[ele] = np.array(data[ele])

        binary_svm = {}

        # for ele in data:
        #     binary_svm[ele] = LinearSVC(random_state=12345)

        pairData = []
        yLabel = np.unique(y)
        for i in range(len(yLabel)):
            for j in range(i + 1, len(yLabel)):
                pairData.append((yLabel[i], yLabel[j]))

        # print('pairdata', pairData)
        for pair in pairData:
            binary_svm[pair] = LinearSVC(random_state=12345)

            xTrain = np.append(data[pair[0]], data[pair[1]], axis=0)
            yTrain1 = np.ones((data[pair[0]].shape[0], 1))
            yTrain2 = np.zeros((data[pair[1]].shape[0], 1))
            yTrain = np.append(yTrain1, yTrain2, axis=0)
            length = yTrain.shape[0]
            yTrain = yTrain.reshape(length, )

            binary_svm[pair].fit(xTrain, yTrain)

        # print('xTrain', xTrain)
        # print('yTrain', yTrain)
        # print('pairData', pairData)
        return binary_svm

    def scores_ovr_student(self, X):
        '''
        Compute class scores for OVR.

        Arguments:
            X: Features to predict.

        Returns:
            scores: a numpy ndarray with scores.
        '''
        # y_pred_train = .predict()
        # print('predict X', X.shape)

        model = self.binary_svm

        scores = []
        for ele in model:
            score = model[ele].decision_function(X)
            # print('score ovr', score.shape)
            scores.append(score)

        scores = np.array(scores).T

        # print('scores', scores)
        return scores

    def scores_ovo_student(self, X):
        '''
        Compute class scores for OVO.

        Arguments:
            X: Features to predict.

        Returns:
            scores: a numpy ndarray with scores.
        '''
        model = self.binary_svm

        scores = np.zeros((X.shape[0], len(model)))
        for ele in model:
            score = model[ele].predict(X)
            for i in range(0, len(score)):
                if score[i] == 1:
                    scores[i][ele[0]] = scores[i][ele[0]] + 1
                else:
                    scores[i][ele[1]] = scores[i][ele[1]] + 1

        # print('scores', scores)
        return scores

    def loss_student(self, W, X, y, C=1.0):
        '''
        Compute loss function given W, X, y.

        For exact definitions, please check the MP document.

        Arugments:
            W: Weights. Numpy array of shape (K, d)
            X: Features. Numpy array of shape (N, d)
            y: Labels. Numpy array of shape N
            C: Penalty constant. Will always be 1 in the MP.

        Returns:
            The value of loss function given W, X and y.
        '''
        K = W.shape[0]
        d = W.shape[1]
        N = X.shape[0]
        # term1 = np.zeros((K,1))

        Term1 = 0
        for i in range(0, K):
            wi = np.linalg.norm(W[i, :])
            term1 = 0.5 * np.power(wi, 2)

            Term1 = Term1 + term1

        svmEquation = 0
        for i in range(0, N):
            term2 = np.zeros((K, 1))
            for j in range(0, K):
                if j == y[i]:
                    deta = 1
                else:
                    deta = 0
                term2[j] = 1 - deta + np.matmul(W[j, :], X[i, :].T)

            maxTerm2 = np.max(term2)
            # finalTerm2 = finalTerm2 + maxTerm2
            term3 = -np.matmul(W[y[i], :], X[i, :].T)
            svmEquation = svmEquation + maxTerm2 + term3

        svmEquation = C * svmEquation + Term1

        return svmEquation

    def grad_student(self, W, X, y, C=1.0):
        '''
        Compute gradient function w.r.t. W given W, X, y.

        For exact definitions, please check the MP document.

        Arugments:
            W: Weights. Numpy array of shape (K, d)
            X: Features. Numpy array of shape (N, d)
            y: Labels. Numpy array of shape N
            C: Penalty constant. Will always be 1 in the MP.

        Returns:
            The graident of loss function w.r.t. W,
            in a numpy array of shape (K, d).
        '''
        K = W.shape[0]
        d = W.shape[1]
        N = X.shape[0]
        result = np.zeros((K, d))
        term1 = W

        result2 = np.zeros((K, d))
        for i in range(0, N):
            term2 = np.zeros((K, 1))

            for j in range(0, K):
                if j == y[i]:
                    deta = 1
                else:
                    deta = 0
                term2[j] = 1 - deta + np.matmul(W[j, :], X[i, :].T)

            indexMaxTerm2 = np.argmax(term2)

            result2[indexMaxTerm2, :] = result2[indexMaxTerm2, :] + X[i, :]
            result2[y[i], :] = result2[y[i], :] - X[i, :]

        result = term1 + result2

        return result
