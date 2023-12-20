import os.path
import pandas as pd
from sklearn.model_selection import train_test_split
def main():
    df = pd.read_csv(root_file)
    df.drop([df.columns[0], 'count', 'hate_speech', 'offensive_language', 'neither', ], axis=1, inplace=True)
    df = df.rename(columns={'class': 'label'})

    # Specify the features (X) and the target variable (y)
    X = df.drop('label', axis=1)  # Assuming 'label' is the target variable
    y = df['label']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Save training data to CSV
    train_data = pd.concat([X_train, y_train], axis=1)
    train_data.to_csv(os.path.join(root, 'train_data.csv'), index=False)

    # Save testing data to CSV
    test_data = pd.concat([X_test, y_test], axis=1)
    test_data.to_csv(os.path.join(root, 'test_data.csv'), index=False)

if __name__ == "__main__":
    root = "data/davidson"
    root_file = os.path.join(root, 'labeled_data.csv')
    main()