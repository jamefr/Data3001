#!/usr/bin/env python3

import pandas as pd
import os
import re
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from pyspark.sql import SparkSession
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel

# My homemade overall accuracy scorer
def overallAccuracy(res1, y_test):
    count = 0
    total = 0
    for k in res1:
        if k == y_test[total]:
            count +=1
        total += 1

# Homemade class-by-class accuracy
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

spark = SparkSession.builder.appName("forest").master("local").getOrCreate()

# training = spark.read.json("training_mouse_computer_room_hub.json").toPandas()
# for file in os.listdir(f"/Users/james/Desktop/DATA3001/Clean_Data"):
#     if re.search("^training", file):
#         print(f"concatenated {file}!")
#         tmp = spark.read.json(file).toPandas()
#         training = pd.concat(objs=[training, tmp])

# test = spark.read.json("test_mouse_computer_room_hub.json").toPandas()
# for file in os.listdir(f"/Users/james/Desktop/DATA3001/Clean_Data"):
#     if re.search("^test", file):
#         print(f"concatenated {file}!")
#         tmp = spark.read.json(file).toPandas()
#         test = pd.concat(objs=[test, tmp])

training = spark.read.json("training_james_5000.json").toPandas()
test = spark.read.json("test_james_5000.json").toPandas()
spark.stop()

# Pop off stop and start times - these are only included for human use
training.pop("flowEndMilliseconds")
training.pop("flowStartMilliseconds")
test.pop("flowEndMilliseconds")
test.pop("flowStartMilliseconds")

# Temp fix
training.pop("firstEightNonEmptyPacketDirections")
test.pop("firstEightNonEmptyPacketDirections")

# Separate the response columns and encode the labels
y_train = training.pop("response")
y_test = test.pop("response")

# Create interaction between ipclassofservice and protocolidentifier

# Create Relevant Dummies
training = pd.get_dummies(training, drop_first=True, columns=["protocolIdentifier"], prefix="protocolIdentifier")
# training = pd.get_dummies(training, columns=["firstEightNonEmptyPacketDirections"], prefix="firstEightNonEmptyPacketDirections")
training = pd.get_dummies(training, drop_first=True, columns=["ipClassOfService"], prefix="ipClassOfService")
training = pd.get_dummies(training, drop_first=True, columns=["flowEndReason"], prefix="flowEndReason")
training = pd.get_dummies(training, drop_first=True, columns=["flowAttributes"], prefix="flowAttributes")

test = pd.get_dummies(test, drop_first=True, columns=["protocolIdentifier"], prefix="protocolIdentifier")
# test = pd.get_dummies(test, columns=["firstEightNonEmptyPacketDirections"], prefix="firstEightNonEmptyPacketDirections")
test = pd.get_dummies(test, drop_first=True, columns=["ipClassOfService"], prefix="ipClassOfService")
test = pd.get_dummies(test, drop_first=True, columns=["flowEndReason"], prefix="flowEndReason")
test = pd.get_dummies(test, drop_first=True, columns=["flowAttributes"], prefix="flowAttributes")

# test = test.sample(frac=0.25, random_state=69)
# test.pop("flowEndReason_eof")
# test.pop("ipClassOfService_0x02")
training.pop("flowAttributes_0a")
# training.pop("ipClassOfService_0xd0")
training.pop("protocolIdentifier_2")

print(training.info())


# corr_matrix = training.corr()
# plt.figure(figsize=(10,10))
# sns.heatmap(corr_matrix, cmap='Blues', annot=True, linecolor='white', linewidth=1)
# plt.show()

# parameters = {"n_estimators":[100,500]}
# tmp = RandomForestClassifier()
# clf = GridSearchCV(tmp, parameters)
# clf = clf.fit(training, y_train)

# Create the random forest model
forest1 = RandomForestClassifier(n_estimators=200, criterion="entropy", random_state=69)
forest1 = forest1.fit(training, y_train)
res1 = forest1.predict(test)
accuracies1 = accuracy_score(y_test, res1)

forest2 = ExtraTreesClassifier(n_estimators=200, criterion="entropy", random_state=69)
forest2 = forest2.fit(training, y_train)
res2 = forest2.predict(test)
accuracies2 = accuracy_score(y_test, res2)

results1 = classByClassAccuracy(res1, y_test)
results2 = classByClassAccuracy(res2, y_test)

print(f"OVERALL ACCURACY\nRF: {accuracies1*100:.6f}\nET: {accuracies2*100:.6f}\n")

print("RANDOM FOREST CLASS BY CLASS ACCURACY")
for k,v in results1.items():
    print(f"class: {k : <40} accuracy: {v:.3%}")

print("\nEXTRA TREES CLASS BY CLASS ACCURACY")
for k,v in results2.items():
    print(f"class: {k : <40} accuracy: {v:.3%}")

print(f"UNIQUE VALUE COUNTS IN Y_PRED")

