"""Grouped Dataset Splitter for ABAC policies.
Ensures that all paraphrased sentences for the same ABAC rule are placed in the same split
to prevent data leakage (overfitting).
"""

from typing import List, Tuple, Dict, Any
from sklearn.model_selection import GroupShuffleSplit
import numpy as np


class GroupedDatasetSplitter:
    """Splits dataset into train/val/test using GroupShuffleSplit."""
    
    def __init__(self, val_size: float = 0.15, test_size: float = 0.15, random_state: int = 42):
        if val_size + test_size >= 1.0:
            raise ValueError("val_size + test_size must be strictly less than 1.0")
        
        self.val_size = val_size
        self.test_size = test_size
        self.random_state = random_state
        
    def split(self, X: List[Any], y: List[Any], groups: List[str]) -> Dict[str, Tuple[List[Any], List[Any]]]:
        """Split data ensuring groups (rule_ids) are not split across sets."""
        # Fast exit if dataset is too small
        unique_groups = len(set(groups))
        if unique_groups < 3:
            raise ValueError(f"Need at least 3 unique groups to perform train/val/test split, got {unique_groups}")
            
        gss_test = GroupShuffleSplit(
            n_splits=1, 
            test_size=self.test_size, 
            random_state=self.random_state
        )
        
        train_val_idx, test_idx = next(gss_test.split(X, y, groups))
        
        X_train_val = [X[i] for i in train_val_idx]
        y_train_val = [y[i] for i in train_val_idx]
        groups_train_val = [groups[i] for i in train_val_idx]
        
        # Calculate validation size relative to train_val dataset
        relative_val_size = self.val_size / (1.0 - self.test_size)
        
        gss_val = GroupShuffleSplit(
            n_splits=1, 
            test_size=relative_val_size, 
            random_state=self.random_state
        )
        
        train_idx, val_idx = next(gss_val.split(X_train_val, y_train_val, groups_train_val))
        
        X_train = [X_train_val[i] for i in train_idx]
        y_train = [y_train_val[i] for i in train_idx]
        
        X_val = [X_train_val[i] for i in val_idx]
        y_val = [y_train_val[i] for i in val_idx]
        
        X_test = [X[i] for i in test_idx]
        y_test = [y[i] for i in test_idx]
        
        return {
            "train": (X_train, y_train),
            "val": (X_val, y_val),
            "test": (X_test, y_test)
        }
