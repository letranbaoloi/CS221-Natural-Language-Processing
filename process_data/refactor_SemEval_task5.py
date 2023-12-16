'''
Format: tsv -> csv
'''
import os.path

import pandas as pd
import csv  # Import the csv module

# define path
ROOT = 'data'
path_test_tsv = 'SemEval_task5/dev_en.tsv'
path_train_tsv = 'SemEval_task5/train_en.tsv'
# Load the TSV file
test = pd.read_csv(os.path.join(ROOT, path_test_tsv), delimiter='\t')
train = pd.read_csv(os.path.join(ROOT, path_train_tsv), delimiter='\t')

# Rename the columns as required
test.rename(columns={'text': 'tweet', 'HS': 'label'}, inplace=True)
train.rename(columns={'text': 'tweet', 'HS': 'label'}, inplace=True)

# Select the columns needed for the new CSV file
test = test[['id', 'tweet', 'label', 'TR', 'AG']]
train = train[['id', 'tweet', 'label', 'TR', 'AG']]
# Write the DataFrame to a CSV file, quoting only the 'tweet' column
# Set the quoting to csv.QUOTE_NONNUMERIC which will quote only non-numeric values
test.to_csv(os.path.join(ROOT, 'SemEval_task5/df_test.csv'), index=False, quoting=csv.QUOTE_NONNUMERIC)
train.to_csv(os.path.join(ROOT, 'SemEval_task5/df_train.csv'), index=False, quoting=csv.QUOTE_NONNUMERIC)