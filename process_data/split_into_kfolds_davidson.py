import os
from sklearn.model_selection import KFold

def split_into_kfolds(df, folds, output_path):
    """Partitions the given data-set into k subsamples (folds). Each fold is intended
    to be used once for testing while the remaining k - 1 folds are used for training.
    This results in k test-sets and k training-sets which are stored as csv files.

    Args:
        df: DataFrame to be split.
        folds: number of folds.
        output_path: path to the directory storing test- and training-sets.
    Return:
        None
    """
    # Check if the directory exists
    if not os.path.exists(output_path):
        # If not, create the directory
        os.makedirs(output_path)

    kfold = KFold(n_splits=folds)
    current_fold = 1

    for train, test in kfold.split(df):
        train_slice = df.iloc[train]
        test_slice = df.iloc[test]
        train_slice.to_csv(os.path.join(output_path, f"train_{current_fold}.csv"), index=False)
        test_slice.to_csv(os.path.join(output_path, f"test_{current_fold}.csv"), index=False)
        current_fold += 1
