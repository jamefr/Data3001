#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
import glob

master = pd.DataFrame()

for x in glob.glob("/Users/james/Desktop/DATA3001/Clean_Data/*.csv"):
    df = pd.read_csv(x)
    master = pd.concat([master, df], ignore_index=True)

set2 = master[master['set'] == 2]
set4 = master[master['set'] == 4]
set6 = master[master['set'] == 6]

# print raw class-byclass precison - text output
# count = 1
# for x in [set2, set4, set6]:
#     for y in x['threshold'].unique():
#         subset = x[x['threshold'] == y]
#         print(f"\nCURRENT SET: {count} CURRENT THRESHOLD: {y}\n")
#         for index, row in subset.iterrows():
#             print(f"Device: {row['devices'] : <40} Total Classifications: {row['total_classifications']}")
#         print()
#     count += 1

# Print overall precision
# counter = 1
# for x in [set2, set4, set6]:
#     print(f"\nCURRENT SET: {counter}\n")
#     for y in x['threshold'].unique():
#         subset = x[x['threshold'] == y]
#         total = 0
#         count = 0
#         for index, row in subset.iterrows():
#             total += row['precison']
#             count += 1
#         print(f"Total Precision: {total/count :.2f}%        thresh{y}")
#     counter += 1

# Show precision vs threshold
# for my_df, name in [(set2, '2 weeks'), (set4, '4 weeks'), (set6, '6 weeks')]:
#     my_df = my_df.drop(['correct_classifications', 'total_classifications', 'set', 'recall'], axis=1)
#     my_df = my_df.iloc[:, 1:]
#     my_df = my_df[~my_df.index.duplicated(keep='first')]
#     my_df = my_df[my_df['devices'] != 'failed']
#     my_df = my_df.sort_values(by='threshold')

#     plt.figure(figsize=(25,7))
#     for device, device_data in my_df.groupby('devices'):
#         plt.plot(device_data['threshold'], device_data['precison'], label=device, )
#     plt.xlabel('Threshold')
#     plt.ylabel('Precision (%)')
#     plt.title(f"{name} / Precision vs. Threshold for Different Devices")
#     plt.legend(loc=4, fontsize=6)
#     plt.show()

# Show Total Classifications vs Threshold
# for my_df, name in [(set2, '2 weeks'), (set4, '4 weeks'), (set6, '6 weeks')]:
#     # Optinally remove a device for missing class testing
#     my_df = my_df[my_df['devices'] == 'candy_house_sesami_wi-fi_access_point']

#     my_df = my_df.sort_values(by='threshold')
#     my_df = my_df.drop(['recall', 'correct_classifications', 'set', 'precison'], axis=1)
#     my_df = my_df.iloc[:, 1:]
#     my_df = my_df[~my_df.index.duplicated(keep='first')]
#     my_df = my_df[my_df['devices'] != 'failed']
    
#     plt.figure(figsize=(25,7))
#     for device, device_data in my_df.groupby('devices'):
#         plt.plot(device_data['threshold'], device_data['total_classifications'], label=device, )
#     plt.ylabel('Total Classifications')
#     plt.xlabel('Threshold')
#     plt.title(f"{name} / Threshold vs. Total Classifications for Different Devices")
#     plt.legend(loc='best', fontsize=6)
#     # plt.xlim(plt.xlim()[::-1])
#     plt.show()

# Show Correct Classifications vs Threshold
# for my_df, name in [(set2, '2 weeks'), (set4, '4 weeks'), (set6, '6 weeks')]:
#     my_df = my_df.sort_values(by='threshold')
#     my_df = my_df.drop(['recall', 'total_classifications', 'set', 'precison'], axis=1)
#     my_df = my_df.iloc[:, 1:]
#     my_df = my_df[~my_df.index.duplicated(keep='first')]
#     my_df = my_df[my_df['devices'] != 'failed']
    
#     plt.figure(figsize=(25,7))
#     for device, device_data in my_df.groupby('devices'):
#         plt.plot(device_data['threshold'], device_data['correct_classifications'], label=device, )
#     plt.ylabel('Correct Classifications')
#     plt.xlabel('Threshold')
#     plt.title(f"{name} / Threshold vs. Correct Classifications for Different Devices")
#     plt.legend(loc='best', fontsize=6)
#     # plt.xlim(plt.xlim()[::-1])
#     plt.show()

# Compare raw model output across all three test sets
# base_set = master[master['threshold'] == 0.0]

# my_df = base_set.drop(['recall', 'total_classifications', 'threshold', 'correct_classifications'], axis=1)
# my_df = my_df.sort_values(by='set')
# my_df = my_df.iloc[:, 1:]
# my_df = my_df[~my_df.index.duplicated(keep='first')]
# my_df = my_df[my_df['devices'] != 'failed']

# plt.figure(figsize=(25,7))
# for device, device_data in my_df.groupby('devices'):
#     plt.plot(device_data['set'], device_data['precison'], label=device, )
# plt.ylabel('Precision')
# plt.xlabel('Weeks Since Training Data')
# plt.title(f"Weeks Away from Training Data vs. Precision for Different Devices")
# plt.legend(loc='best', fontsize=6)
# # plt.xlim(plt.xlim()[::-1])
# plt.show()


# Priyash bar chart
ind = np.arange(25)
width = 0.5

plt.figure(figsize=(10,10))
my_df = set2[set2['devices'] != 'failed']
my_df = my_df[my_df['threshold'] == 0.8]
categories = my_df['devices']
counts = my_df['precison']


# Sort the data by counts (and keep the categories aligned)
sorted_categories = [x for _, x in sorted(zip(counts, categories))]
sorted_counts = sorted(counts)

# Create horizontal bar chart with increased bar width
plt.barh(sorted_categories, sorted_counts, color='#000b3e', label='80% Threshold')  # Adjust 'height' for thicker bars 

# Add text for counts as integers
for index, value in enumerate(sorted_counts):
    plt.text(value, index, f"{value:.0f}")  # Formats the number as an integer

# Add labels and title
plt.legend(loc=4)
plt.xlabel('Number of classificaitons')
plt.ylabel('Device')
plt.title('Class by class precision for 0-2 weeks test set')
plt.tight_layout()
plt.show()


