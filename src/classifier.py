import numpy as np

class Classifier:
    def __init__(self):
        pass
    def train(self, X_train, Y_train, X_dev, Y_dev):
        raise NotImplementedError
    def get_weights(self):
        raise NotImplementedError

class SKlearnClassifier(Classifier):
    def __init__(self, m):
        self.model = m
    def train(self, X_train, Y_train, X_dev, Y_dev):
        self.model.fit(X_train, Y_train)
        score = self.model.score(X_dev, Y_dev)
        return score
    def get_weights(self):
        w = self.model.coef_
        if len(w.shape) == 1:
                w = np.expand_dims(w, 0)
        return w

