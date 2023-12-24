'''
Format: tsv -> csv
'''
import os.path

import pandas as pd
import csv  # Import the csv module

# define path
ROOT = 'data'
path_test_tsv = 'SemEval_task5/df_test_org.csv'
path_train_tsv = 'SemEval_task5/train_en.tsv'
path_dev_tsv = 'SemEval_task5/dev_en.tsv'
# Load the TSV file
test = pd.read_csv(os.path.join(ROOT, path_test_tsv))
train = pd.read_csv(os.path.join(ROOT, path_train_tsv), delimiter='\t')
dev = pd.read_csv(os.path.join(ROOT, path_dev_tsv), delimiter='\t')

# Rename the columns as required
test.rename(columns={'task_idx': 'id'}, inplace=True)
train.rename(columns={'text': 'tweet', 'HS': 'label'}, inplace=True)
dev.rename(columns={'text': 'tweet', 'HS': 'label'}, inplace=True)
train_dev = pd.concat([train, dev], ignore_index=True)
# Select the columns needed for the new CSV file
test = test[['tweet', 'label']]
train_dev = train_dev[['tweet', 'label']]
# Write the DataFrame to a CSV file, quoting only the 'tweet' column
# Set the quoting to csv.QUOTE_NONNUMERIC which will quote only non-numeric values
test.to_csv(os.path.join(ROOT, 'SemEval_task5/df_test.csv'), index=False, quoting=csv.QUOTE_NONNUMERIC)
train_dev.to_csv(os.path.join(ROOT, 'SemEval_task5/df_train_dev.csv'), index=False, quoting=csv.QUOTE_NONNUMERIC)