import os
import random
from sklearn.model_selection import train_test_split, KFold
import shutil

class DataSplitter:
    """
    This class provides multiple methods to split datasets for training and evaluation.
    """

    def __init__(self, data_dir="data/ocr_processed", output_dir="data/splits"):
        """
        Initializes the DataSplitter with input and output directories.

        :param data_dir: Directory containing the processed text data.
        :param output_dir: Directory to save the split data.
        """
        self.data_dir = data_dir
        self.output_dir = output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    def list_files(self):
        """
        Lists all files in the dataset directory.
        
        :return: A list of file paths.
        """
        return [os.path.join(self.data_dir, f) for f in os.listdir(self.data_dir)]

    def train_test_split(self, test_size=0.2, random_state=42):
        """
        Splits the dataset into training and testing sets.

        :param test_size: Proportion of the dataset to include in the test split.
        :param random_state: Seed for shuffling.
        """
        files = self.list_files()
        train_files, test_files = train_test_split(files, test_size=test_size, random_state=random_state)

        self._save_split(train_files, "train")
        self._save_split(test_files, "test")

    def k_fold_split(self, k=5):
        """
        Performs K-Fold cross-validation splitting.

        :param k: Number of folds for cross-validation.
        """
        files = self.list_files()
        kf = KFold(n_splits=k, shuffle=True)

        for fold, (train_idx, test_idx) in enumerate(kf.split(files)):
            train_files = [files[i] for i in train_idx]
            test_files = [files[i] for i in test_idx]
            fold_dir = os.path.join(self.output_dir, f"fold_{fold+1}")
            self._save_split(train_files, "train", fold_dir)
            self._save_split(test_files, "test", fold_dir)

    def _save_split(self, files, split_name, fold_dir=None):
        """
        Saves the split files to the appropriate directory.
        
        :param files: List of file paths to save.
        :param split_name: Either 'train' or 'test'.
        :param fold_dir: Directory for saving k-fold splits.
        """
        split_dir = fold_dir if fold_dir else self.output_dir
        save_dir = os.path.join(split_dir, split_name)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        for file in files:
            shutil.copy(file, save_dir)

if __name__ == "__main__":
    splitter = DataSplitter()

    # Example 1: Train-test split
    splitter.train_test_split(test_size=0.2)

    # Example 2: K-fold cross-validation (5 folds)
    splitter.k_fold_split(k=5)
