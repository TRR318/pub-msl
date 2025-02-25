import numpy as np
from scipy.special import logsumexp
from sklearn.base import ClassifierMixin, BaseEstimator
from sklearn.naive_bayes import BernoulliNB


class StagedNBClassifier(ClassifierMixin, BaseEstimator):
    def __init__(self):
        self.clf = None

    def fit(self, X, y):
        self.clf = BernoulliNB().fit(X, y)
        return self

    def __getitem__(self, item):
        return FeatureSubsetNBClassifier(self, np.array(list(set(item))))


class FeatureSubsetNBClassifier(ClassifierMixin, BaseEstimator):
    def __init__(self, parent: StagedNBClassifier, feature_idx):
        self.parent = parent
        self.feature_idx = feature_idx
        self.clf = parent.clf

        self.classes_ = self.clf.classes_

    def predict_proba(self, X):
        if self.feature_idx.size > 0:
            X = X[:, self.feature_idx]
            feature_log_proba = self.clf.feature_log_prob_[:, self.feature_idx]
            # Compute  neg_prob · (1 - X).T  as  ∑neg_prob - X · neg_prob
            neg_prob = np.log(1 - np.exp(feature_log_proba))
            jll = X @ (feature_log_proba - neg_prob).T
            jll += self.clf.class_log_prior_ + neg_prob.sum(axis=1)
            # normalize by P(x) = P(f_1, ..., f_n)
            log_prob_x = logsumexp(jll, axis=1)
            logproba = jll - np.atleast_2d(log_prob_x).T
        else:
            logproba = np.repeat([self.clf.class_log_prior_], X.shape[0], axis=0)

        # predict proba
        return np.exp(logproba)

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)


if __name__ == '__main__':
    from sklearn.datasets import load_iris
    from skpsl.preprocessing import MinEntropyBinarizer

    X, y = load_iris(return_X_y=True)
    X = MinEntropyBinarizer().fit_transform(X, y)
    clf = StagedNBClassifier().fit(X, y)
    print(clf[{3, 1}].predict_proba(X))
