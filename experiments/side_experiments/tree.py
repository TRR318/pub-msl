import numpy as np
from sklearn.tree import DecisionTreeClassifier


# Custom function to predict with restricted features
def custom_predict(tree, X_instance, available_features):
    """
    Recursively traverse the decision tree for a single instance and make a prediction
    while skipping unavailable features.

    :param tree: Trained decision tree (tree_ attribute)
    :param X_instance: The test instance (1D array)
    :param available_features: The list of features available for this instance
    :return: The predicted class label
    """
    node = 0  # Start at the root node

    while tree.feature[node] != -2:  # While not a leaf node (-2 indicates a leaf)
        feature = tree.feature[node]
        threshold = tree.threshold[node]

        if (
            feature in available_features
        ):  # If the feature is available, follow the split
            if X_instance[feature] <= threshold:
                node = tree.children_left[node]
            else:
                node = tree.children_right[node]
        else:  # If the feature is not available, make a "guess"
            # Heuristic: Go down the most common path (or randomly choose a path)
            node = tree.children_left[
                node
            ]  # You can adjust this based on your preference

    # Return the predicted class (leaf value)
    return np.argmax(tree.value[node])


# Example usage
# Train a Decision Tree
X_train = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
y_train = np.array([0, 1, 0, 1])

clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)

# Access the trained tree structure
tree = clf.tree_

# Test instance (full feature vector)
X_test = np.array([6, 7])

# List of available features for this instance (only feature 0 is available)
available_features = [1]  # Suppose feature 1 is missing

# Predict using custom logic
prediction = custom_predict(tree, X_test, available_features)
print(f"Prediction with available features {available_features}: {prediction}")
