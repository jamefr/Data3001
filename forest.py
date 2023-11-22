#!/usr/bin/env python3

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score


def classByClassAccuracy(predictions, actual):
    ind = 0
    classes = {}
    for i in predictions:
        if i in classes.keys():
            if i == actual[ind]:
                classes[i]["count"] += 1
            classes[i]["total"] += 1
        else:
            if i == actual[ind]:
                classes[i] = {"count":1, "total":1}
            else:
                classes[i] = {"count":0, "total":1}
        
        ind += 1

    results = {}
    for k in classes.keys():
        results[k] = (classes[k]["count"] , classes[k]["total"])

    return dict(sorted(results.items(), key=lambda x:x[1], reverse=True))

# Converting Variables into category
train_df = pd.read_json('train_df_37.json', lines=True)
train_df = train_df.replace(np.nan, 0)
test_df = pd.read_json('test_df_week2.json', lines=True)

columns_to_remove = ["flowStartMilliseconds", "flowEndMilliseconds",'firstEightNonEmptyPacketDirections', 'destinationTransportPort']
for regressors in columns_to_remove:
    if regressors in train_df.columns:
        train_df = train_df.drop(columns=regressors)
        test_df = test_df.drop(columns=regressors)

columns_categorical = ["flowAttributes", "protocolIdentifier", "ipClassOfService", "flowEndReason", 'reverseFlowAttributes']

for regressors in columns_categorical:
    train_df = pd.get_dummies(train_df, columns=[regressors], prefix=regressors)
    test_df = pd.get_dummies(test_df, columns=[regressors], prefix=regressors)

for regressors in test_df.columns:
    if regressors not in train_df.columns:
         test_df = test_df.drop(columns=regressors)

# Manually
category = ['ipClassOfService_0xd0']
for regressors in category:
    if regressors in train_df.columns:
        train_df = train_df.drop(columns=regressors)
    if regressors in test_df.columns:
        test_df = test_df.drop(columns=regressors)


response = ['response']
predictors = [x for x in list(train_df.columns) if x not in response]

# predictors.remove('ipClassOfService_4')
X = train_df[predictors]
y = np.ravel(train_df[response])
X_test = test_df[predictors]
y_test = np.ravel(test_df[response])
forest1 = RandomForestClassifier(criterion='entropy', bootstrap=True)

param_grid2 = [{"n_estimators" : [200,400]}, {"class_weight" : ["balanced_subsample", "balanced"]}]
forestSearch = GridSearchCV(forest1, param_grid=param_grid2, scoring='accuracy', cv=3, verbose=1, n_jobs=-1)

forestFit = forestSearch.fit(X,y)
best_forest = forestFit.best_estimator_

y_pred = best_forest.predict(X_test)

print(accuracy_score(y_test, y_pred))

# Use threshold to increase precision
# y_test_arr = np.ravel(y_test)
# probabilities = best_forest.predict_proba(X_test)

# prob = pd.DataFrame(probabilities)
# mean_prob = np.ravel(prob.mean(axis=0))
# std_deviation = np.ravel(prob.std(axis=0))
# mean_reduced_prob = (probabilities - mean_prob) / std_deviation

# z_score = (0.35 - mean_prob) / std_deviation
# threshold = mean_reduced_prob > z_score
# predictions = np.full((probabilities.shape[0],), -1)
# for i, instance in enumerate(threshold):
#     # Check if any probability meets the threshold
#     if any(instance):
#         # Get the index of the max probability above the threshold
#         predictions[i] = np.argmax(mean_reduced_prob[i])

# accuracy = classByClassAccuracy(predictions, y_test_arr)
# print(predictions)