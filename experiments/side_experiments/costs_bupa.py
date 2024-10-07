import pandas as pd
from sklearn.model_selection import train_test_split
from skpsl import ProbabilisticScoringList

from sklearn.metrics import accuracy_score, brier_score_loss, balanced_accuracy_score

from experiments.util import DataLoader
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline


from sklearn.tree import DecisionTreeClassifier

# pipeline_dt = make_pipeline(
#     SimpleImputer(missing_values=-1, strategy="most_frequent"),
#     DecisionTreeClassifier(),
# )

# pipeline_dt.fit(X_train, y_train)
# imputer = pipeline_dt[0]
# dt = pipeline_dt[1]


import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris

def get_bupa_cost_for_feature_subset(subset):
    cost = 0.0
    unique_values, indices = np.unique(subset, return_index=True)
    unique_in_order = unique_values[np.argsort(indices)]
    if unique_in_order.size >= 0.0:
        for i,s in enumerate(unique_in_order):
            if i == 0:
                cost += costs_bupa.iloc[s,0]
            else:
                cost += costs_bupa.iloc[s,1]
        return cost
    else:
        raise TypeError("Alarm")

def predict_with_cost(dt, X_test):
    y_probas = []
    features_pred = []
    cost_pred = []

    for x_test in X_test:
        x_test = x_test.reshape(1,-1)
        y_proba = dt.predict_proba(x_test)
        y_probas.append(y_proba)

        # Get the decision path for the test instance
        decision_path = dt.decision_path(x_test)

        # Get feature indices used at each node
        node_indicator = decision_path.indices
        features_used = dt.tree_.feature[node_indicator]

        # Filter out any -2s, which represent leaf nodes
        features_used = features_used[features_used != -2]
        
        features_pred.append(features_used)
        cost_summand = get_bupa_cost_for_feature_subset(features_used)
        cost_pred.append(cost_summand)
    return np.asarray(y_proba), cost_pred, features_pred 

import numpy as np

def limited_feature_predict(clf, X, max_features):
    """
    Make predictions by limiting the number of features the tree can make.
    
    Parameters:
    clf: Trained decision tree classifier
    X: Feature matrix
    max_splits: Maximum number of features to traverse before making a prediction
    
    Returns:
    predictions: Array of predictions for each sample in X
    """
    tree = clf.tree_
    n_samples = X.shape[0]
    predictions = []
   
    feature_lists = []

    for i in range(n_samples):
        feature_list = []
        node_id = 0  # Start at the root node

        while tree.children_left[node_id] != tree.children_right[node_id]:  # While not a leaf

            # Get the feature to split on and the threshold to compare to
            feature = tree.feature[node_id]

            # Stop if the max number of splits is reached
            if len(set(feature_list + [feature])) > max_features:
                break
            
            feature_list.append(feature)
            threshold = tree.threshold[node_id]

            # Traverse to the left or right child based on feature value
            if X[i, feature] <= threshold:
                node_id = tree.children_left[node_id]
            else:
                node_id = tree.children_right[node_id]

        # After the loop, we are at a node, make a prediction
        # Use the majority class of this node to make a prediction
        predictions.append(tree.value[node_id]) 
        feature_lists.append(feature_list)

    return np.vstack(predictions), feature_lists



if __name__ == "__main__":

    data_bupa = pd.read_csv("experiments/data/liver_disorders/bupa.data", sep=",", header=None, names = ["mcv",
    "alkphos",
    "sgpt",
    "sgot",
    "gammagt",
    "drinks",
    "selector"])

    data_bupa= data_bupa.drop(columns=["selector"])
    data_bupa["drinks"] = (data_bupa["drinks"] >= 3).astype("int")

    costs_bupa = pd.read_csv("experiments/data/liver_disorders/bupa-liver.expense", sep=r"\t+", header = None, index_col=0, engine="python")
    costs_bupa[1] = costs_bupa[1].str.strip(",").astype(float)
    costs_bupa.index = costs_bupa.index.str.strip(":")


    score_set = {-3,-2,-1,1,2,3}



    for seed in range(0,100):

        X, y = data_bupa.iloc[:,:-1].to_numpy(), data_bupa.iloc[:,-1].to_numpy() 

        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=seed)

        dt = DecisionTreeClassifier().fit(X_train, y_train)
        psl = ProbabilisticScoringList(score_set=score_set).fit(X_train, y_train)

        num_features = X.shape[1]

        briers_psl = []
        briers_dt = []

        costs_psl = []
        costs_dt = []

        for i in range(0,num_features):
            y_prob_psl = psl.predict_proba(X_test, k=i)

            print(psl[i].features)
            psl_cost = len(X_test) * get_bupa_cost_for_feature_subset(psl[i].features) 
            print("psl:", psl[i].features)

            y_prob_dt, used_features = limited_feature_predict(dt, X_test, max_features=i)

            dt_cost = 0

            for feature_subset in used_features:
                dt_cost += get_bupa_cost_for_feature_subset(feature_subset)

            print("dt:", len(used_features), dt_cost)

            briers_psl.append(brier_score_loss(y_test, np.argmax(y_prob_psl,axis=1)))
            briers_dt.append(brier_score_loss(y_test, np.argmax(y_prob_dt,axis=1)))

            costs_psl.append(psl_cost)
            costs_dt.append(dt_cost)
