#!/usr/bin/env python3

import re
import sys
import pandas as pd
from pyspark.sql import SparkSession
import pyspark.sql.functions as F

spark = SparkSession.builder.appName("cleanData").getOrCreate()

""" These are dodgy for these reasons:
    - Only one value across all records (silkapplable, initialTCPFlags, egressInterface, ingressInterface, observationDomainId, CollectorName)
    - Need to omit for validity (Mac addresses, IP addresses)
    - Direction of travel is reversed
    - So many missing values that we were left with insufficient data to run the analysis (reverseStandardDeviationInterarrivalTime, reverseStandardDeviationPayloadLength, "reverseSmallPacketCount", "reverseNonEmptyPacketCount", "reverseMaxPacketSize", "reverseLargePacketCount", "reverseFirstNonEmptyPacketSize", "reverseDataByteCount", "reverseBytesPerPacket", "reverseAverageInterarrivalTime")
"""
dodgy_regressors = ["destinationMacAddress", "sourceMacAddress", "egressInterface", "ingressInterface", "initialTCPFlags", 
                    "reverseInitialTCPFlags", "reverseTcpUrgTotalCount", "reverseUnionTCPFlags", "silkAppLabel", 
                    "tcpSequenceNumber", "tcpUrgTotalCount", "unionTCPFlags", "vlanId", "sourceIPv4Address", 
                    "destinationIPv4Address", "reverseTcpSequenceNumber", "observationDomainId", "collectorName"]

impute_list = {"reverseStandardDeviationInterarrivalTime":0, "reverseStandardDeviationPayloadLength":0, 
               "reverseSmallPacketCount":0, "reverseNonEmptyPacketCount":0, "reverseMaxPacketSize":0, 
               "reverseLargePacketCount":0, "reverseFirstNonEmptyPacketSize":0, "reverseDataByteCount":0, 
               "reverseBytesPerPacket":0, "reverseAverageInterarrivalTime":0}

# na_susbet includes all regressors we need to ensure have no null values
na_subset = ["protocolIdentifier"]

df1 = spark.read.json(path=sys.argv[1]).select(F.col('flows.*'))
df = df1.sort(F.col("flowEndMilliseconds")).toPandas()

# Make sample of 5000 lines, based upon flowStartMilliseconds
count = 0
df_training = pd.DataFrame()
for x in df:
    date = df['flowEndMilliseconds'][count]
    if date == None:
        continue
    if date <= "2019-08-26":
        df_training.append(x)
    else:
        break
    count += 1

count = 0
df_test = pd.DataFrame()
for x in df:
    date = df['flowEndMilliseconds'][count]
    if date == None:
        continue
    if date > "2019-08-27" and date < "2019-09-10":
        df_training.append(x)
    else:
        break
    count += 1

# Remove problematic regressors:
for dodgy in dodgy_regressors:
    df_training = df_training.drop(dodgy, axis=1)
    df_test = df_test.drop(dodgy, axis=1)

# Impute the impute list
df_training = df_training.fillna(value=impute_list)
df_test = df_test.fillna(value=impute_list)

# Add response variable
response = re.sub('([^.]*).json$', r'\1', sys.argv[1])
df_training = df_training.withColumn("response", F.lit(response))
df_test = df_test.withColumn("response", F.lit(response))

# Create Pandas DataFrame and clear any remaining null values
for item in na_subset:  
    df_training = df_training.na.drop('all', subset=[item])
    df_test = df_test.na.drop('all', subset=[item])

df_trainning = df_training.toPandas()
df_test = df_test.toPandas()

df_training.dropna(axis=0, inplace=True)
df_test.dropna(axis=0, inplace=True)

for columns in df_training:
    print(f"column: {columns} {df_training[columns].unique()}\n")
print(len(df_training))

# Save to file
df_training.to_json(rf"~/Desktop/DATA3001/Clean_Data/training_{sys.argv[1]}", orient='records', lines=True)
df_test.to_json(rf"~/Desktop/DATA3001/Clean_Data/test_{sys.argv[1]}", orient='records', lines=True)