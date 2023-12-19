import os.path

import pandas as pd
from split_into_kfolds_davidson import split_into_kfolds
def main():
    df = pd.read_csv(root_file)
    df.drop([df.columns[0], 'count', 'hate_speech', 'offensive_language', 'neither', ], axis=1, inplace=True)
    df = df.rename(columns={'class': 'label'})
    split_into_kfolds(df, 5, output_path)

if __name__ == "__main__":
    root = "data/davidson"
    root_file = os.path.join(root, 'labeled_data.csv')
    output_path = os.path.join(root, 'kfolds')
    main()