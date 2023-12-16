import csv
import os.path

import pandas as pd
from sklearn.model_selection import train_test_split
import re

root = "data/davidson"
csv_file_path = os.path.join(root, "labeled_data.csv")
txt_file_path = os.path.join(root, "labeled_data.txt")

csv_dv_data = os.path.join(root, "dv_data.csv") # save file dv_data

unfiltered = [line.rstrip('\n') for line in open('data/davidson/davidson.txt')]
pattern = re.compile(r'^\d+,')  # ^\d+ matches one or more digits at the beginning of a line, followed by a comma
filtered = [line for line in unfiltered if line and pattern.match(line)]

print(len(filtered))
dv = ['id,tweet,label']
with open(csv_dv_data, 'w') as fp:
    for line in filtered:
        parts = line.split(",")
        tweet = ', '.join(parts[6:]).replace("\"", "")
        fp.write("%s\n" % (f"{parts[0]},\"{tweet}\",{parts[5]}"))

# # Load the dataset
# df = pd.read_csv(csv_dv_data)
#
# # Split the dataset into training and testing sets
# train, test = train_test_split(df, test_size=0.1, random_state=42)
#
# # Save the training and testing sets into separate CSV files
# train.to_csv(os.path.join(root, 'dv_train.csv'), index=False)
# test.to_csv(os.path.join(root, 'dv_test.csv'), index=False)