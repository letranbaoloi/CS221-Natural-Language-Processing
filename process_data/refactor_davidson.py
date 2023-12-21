import os.path
import pandas as pd
from sklearn.model_selection import train_test_split
import csv
def main():
    df = pd.read_csv(root_file)
    df.drop([df.columns[0], 'count', 'hate_speech', 'offensive_language', 'neither', ], axis=1, inplace=True)
    df = df.rename(columns={'class': 'label'})

    hate_samples = df[df['label'] == 1]
    non_hate_samples = df[df['label'] == 0]
    hate_train = hate_samples.sample(frac=0.8, random_state=42)
    hate_test = hate_samples.drop(hate_train.index)

    non_hate_train = non_hate_samples.sample(frac=0.8, random_state=42)
    non_hate_test = non_hate_samples.drop(non_hate_train.index)

    # Concatenate hate and non-hate samples for train and test sets
    train_set = pd.concat([hate_train, non_hate_train])
    test_set = pd.concat([hate_test, non_hate_test])

    # Shuffle the datasets
    train_set = train_set.sample(frac=1, random_state=42).reset_index(drop=True)
    test_set = test_set.sample(frac=1, random_state=42).reset_index(drop=True)

    # Write the DataFrames to CSV files, quoting only the 'tweet' column
    # Set the quoting to csv.QUOTE_NONNUMERIC which will quote only non-numeric values
    train_set.to_csv(os.path.join(ROOT, 'train_data.csv'), index=False, quoting=csv.QUOTE_NONNUMERIC)
    test_set.to_csv(os.path.join(ROOT, 'test_data.csv'), index=False, quoting=csv.QUOTE_NONNUMERIC)

if __name__ == "__main__":
    ROOT = "data/davidson"
    root_file = os.path.join(ROOT, 'labeled_data.csv')
    main()