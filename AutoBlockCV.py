from sklearn.model_selection import BaseCrossValidator
import numpy as np
class AutoBlockCV(BaseCrossValidator):
    def __init__(self, min_index, n_splits=5):
        """
        Automatically splits data into non-overlapping test sets after min_index.
        :param min_index: The index after which test sets begin.
        :param n_splits: Number of test splits (default=5).
        """
        self.min_index = min_index
        self.n_splits = n_splits

    def split(self, X, y=None, groups=None):
        n_samples = len(X)
        test_size = (n_samples - self.min_index) // self.n_splits  # Evenly divide remaining data

        for i in range(self.n_splits):
            start = self.min_index + i * test_size
            end = min(start + test_size, n_samples)  # Avoid overflow
            test_index = np.arange(start, end)
            train_index = np.setdiff1d(np.arange(n_samples), test_index)
            yield train_index, test_index

    def get_n_splits(self, X, y=None, groups=None):
        return self.n_splits