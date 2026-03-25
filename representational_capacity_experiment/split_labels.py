import pandas as pd
import numpy as np

def split_labels(dataset: pd.DataFrame, factorize=False):
    if factorize:
        dataset = dataset.copy()
        for col in dataset:
            if dataset[col].dtype == object:
                dataset[col] = pd.factorize(dataset[col].values)[0]

    possible_label_col = None
    y_candidate = None
    are_labels = False
    for col in dataset.columns[::-1]:
        possible_label_col = col
        y_candidate = dataset[possible_label_col].values
        are_labels = (possible_label_col.lower() in ['label', 'labels', 'target', 'class']) \
        or (
            (
                np.issubdtype(y_candidate.dtype, np.integer)
                or np.issubdtype(y_candidate.dtype, np.object_)
            )
            and not any(
                np.issubdtype(y_candidate.dtype, dtype)
                for dtype in [np.datetime64, np.floating, np.bool]
            ) 
            and possible_label_col.lower() not in ["date", "time", "datetime"]
            and len(np.unique(y_candidate)) <= .1*dataset.shape[0] 
        )
        if are_labels:
            break
    
    if not are_labels:
        raise ValueError("Could not find label column")
    
    return dataset.drop(columns=[possible_label_col]).values, y_candidate, possible_label_col