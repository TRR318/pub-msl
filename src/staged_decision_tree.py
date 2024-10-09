import numpy as np
from sklearn.base import ClassifierMixin
from sklearn.tree import DecisionTreeClassifier


class StagedDecisionTreeClassifier(DecisionTreeClassifier):
    def __iter__(self):
        for i in range(self.tree_.n_features):
            yield MaxFeatureDecisionTreeClassifier(self, i)


class MaxFeatureDecisionTreeClassifier(ClassifierMixin):
    def __init__(self, clf, max_features):
        self.dt = clf
        self.max_features_ = max_features
        self.classes_ = clf.classes_

    def predict(self, X, check_input=True):
        return np.argmax(self.dt.predict_proba(X, check_input), axis=1)

    def predict_proba(self, X, check_input=True):
        tree = self.dt.tree_
        predictions = []

        for row in X:
            feature_list = []
            node_id = 0  # Start at the root node

            while tree.children_left[node_id] != tree.children_right[node_id]:  # While not a leaf
                # Get the feature to split on and the threshold to compare to
                f = tree.feature[node_id]

                # Stop if the max number of splits is reached
                if len(set(feature_list + [f])) > self.max_features_:
                    break

                feature_list.append(f)
                t = tree.threshold[node_id]

                # Traverse to the left or right child based on feature value
                node_id = tree.children_left[node_id] if row[f] <= t else tree.children_right[node_id]

            # After the loop, we are at a node, make a prediction
            # Use the majority class of this node to make a prediction
            predictions.append(tree.value[node_id])
        return np.vstack(predictions)
