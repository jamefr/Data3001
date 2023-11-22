#!/usr/bin/env python3

import pandas as pd
import numpy as np
import re
import sys
import xgboost as xgb
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
import glob

train_df = pd.read_json('train_df_37.json', lines=True)
train_df = train_df.replace(np.nan, 0)
test_df = pd.read_json(sys.argv[1], lines=True)
devname, digit, _ = re.split(r"([0-9])", sys.argv[1])

def classByClassAccuracy(predictions, actual):
    ind = 0
    classes = {}
    for j in predictions:
        if j == -1:
            i = 'failed'
        else: 
            i = decoded_category[j]
        if i in classes.keys():
            
            if i == decoded_category[actual[ind]]:
                classes[i]["count"] += 1
            classes[i]["total"] += 1
        else:
            
            if i == decoded_category[actual[ind]]:
                classes[i] = {"count":1, "total":1}
            else:
                classes[i] = {"count":0, "total":1}
        
        ind += 1
 
    results = {}
    for k in classes.keys():
        results[k] = (classes[k]["count"], classes[k]["total"])
 
    return dict(sorted(results.items(), key=lambda x:x[1], reverse=True))

columns_to_remove = ["flowStartMilliseconds", "flowEndMilliseconds",'firstEightNonEmptyPacketDirections', 'destinationTransportPort']
for regressors in columns_to_remove:
    if regressors in train_df.columns:
        train_df = train_df.drop(columns=regressors)
        test_df = test_df.drop(columns=regressors)

# Encoding the response to numeric values
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
train_df['response'] = le.fit_transform(train_df['response'])
test_df['response'] = le.fit_transform(test_df['response'])


# Optionally omit classes from test
# test_df = test_df[test_df['response'] != 2]

# Converting Variables into category
columns_categorical = ["flowAttributes", "protocolIdentifier", "ipClassOfService", "flowEndReason",
                      'reverseFlowAttributes', 'sourceTransportPort']

for regressors in columns_categorical:
    train_df[regressors] = train_df[regressors].astype('category')
    test_df[regressors] = test_df[regressors].astype('category')

for col in columns_categorical:
    train_df[col] = train_df[col].cat.codes
    test_df[col] = test_df[col].cat.codes

response = ['response']
predictors = [x for x in list(train_df.columns) if x not in response]

X = train_df[predictors]
y = train_df[response]
X_test = test_df[predictors]
y_test = test_df[response]

# hyperparameter train XGBoost
param_grid = {
    'n_estimators': [50, 100],
    'learning_rate': [0.1, 0.5],
    'max_depth': [5, 10],
    'alpha': [0.1, 1]
}
model = xgb.XGBClassifier(use_label_encoder=False)
grid_search = GridSearchCV(model, param_grid, scoring='accuracy', cv=2, verbose=1, n_jobs=-1)

model = xgb.XGBClassifier(alpha=1, learning_rate=0.1, max_depth=5, n_estimators=50)
model.fit(X, y)

# Evaluate the model
y_pred = model.predict(X_test)
print("Test accuracy:", accuracy_score(y_test, y_pred))

# Use threshold to increase precision
y_test_arr = np.ravel(y_test)
probabilities = model.predict_proba(X_test)

prob = pd.DataFrame(probabilities)
mean_prob = np.ravel(prob.mean(axis=0))
std_deviation = np.ravel(prob.std(axis=0))
mean_reduced_prob = (probabilities - mean_prob) / std_deviation

print()

for x in [0, 0.2, 0.4, 0.6, 0.8, 1]:
    z_score = (x - mean_prob) / std_deviation
    threshold = mean_reduced_prob > z_score
    predictions = np.full((probabilities.shape[0],), -1)
    for i, instance in enumerate(threshold):
        # Check if any probability meets the threshold
        if any(instance):
            # Get the index of the max probability above the threshold
            predictions[i] = np.argmax(mean_reduced_prob[i])

    conversion = np.arange(0,25)
    decoded_category = le.inverse_transform(conversion)
    accuracy = classByClassAccuracy(predictions, y_test_arr)
    # print(accuracy)

    row_list = []
    for k,v in accuracy.items():
        tmp = {'devices':k, 'correct_classifications':v[0], 'total_classifications':v[1], 'precison':v[0]/v[1]*100, 'recall':v[0]/1000, 'set':digit ,'threshold':x}
        row_list.append(tmp)
        print(f"Device: {k : <40} Precision: {(v[0]/v[1])*100 :.2f}%")

    res = pd.DataFrame(row_list)
    print(res, res.shape)
    # res.to_csv(rf"~/Desktop/DATA3001/Clean_Data/{devname}{digit}_results_{x}.csv")

print()

# # Using seaborn, produce a colour map to show which covariates have high correlation #
# sns.set(rc={'font.size': 6})
# corr_matrix = train_df.corr()
# print(corr_matrix)
# plt.figure(figsize=(10,10))
# sns.heatmap(corr_matrix, cmap='Blues', annot=True, linecolor='white', linewidth=1)
# plt.tight_layout()
# plt.show()