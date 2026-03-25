import numpy as np
from scipy import stats

def confidence_interval(data, confidence=0.95)->tuple[float, tuple[float, float]]:
    a = np.array(data)
    n = len(a)
    if n < 2:
        raise ValueError("At least two data points are required to compute a confidence interval.")
    
    mean = np.mean(a)
    sem = stats.sem(a) 
    margin = sem * stats.t.ppf((1 + confidence) / 2.0, df=n-1) 
    return float(margin), (float(mean - margin), float(mean + margin))