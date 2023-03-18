import pandas as pd
import numpy as np
import json

def recall(ground_truth, preds):
    return np.sum(np.logical_and(ground_truth == 1, preds == 1)) / np.sum(ground_truth==1)
def precision(ground_truth, preds):
    return np.sum(np.logical_and(ground_truth == 1, preds == 1)) / np.sum(preds==1)
def top_p(distribution, p=0.80):
    """
    Return ground truth
    """
    sorted_indices = np.argsort(distribution[::-1])
    distribution = distribution[sorted_indices]
    cumsum_distrib = np.cumsum(distribution)
    indices = sorted_indices[cumsum_distrib <= p]
    
    preds = np.zeros(distribution.shape[0])
    preds[indices] = 1
    return preds
def top_k(distribution, k=20):
    """
    Return ground truth
    """
    sorted_indices = np.argsort(distribution[::-1])
    distribution = distribution[sorted_indices]
    cumsum_distrib = np.cumsum(distribution)
    indices = sorted_indices[:k+1]
    
    preds = np.zeros(distribution.shape[0])
    preds[indices] = 1
    return preds
def eps(distribution, eps=0.1):
    """
    Return ground truth
    """
    preds = np.zeros(distribution.shape[0])
    preds[distribution > eps] = 1
    return preds


data = pd.read_csv('./eval_data.csv')
distribution_types = ["cosine_mean", "cosine_max", "cosine_sent"]
sampling_functions = [top_p, top_k, eps]
values = [
    ((np.arange(9)+1)*0.1).tolist(),
    [10, 20, 30, 40, 50],
    [0.1, 0.05, 0.02, 0.01, 0.001]
]
ground_truth = np.array(data['Ground truth label'])

results = dict()

for distrib_type in distribution_types:
    scores = np.array(data[distrib_type])
    distribution = np.exp(scores) / np.sum(np.exp(scores))

    if distrib_type not in results:
        results[distrib_type] = dict()
    for func, vals in zip(sampling_functions, values):
        if func.__name__ not in results[distrib_type]:
            results[distrib_type][func.__name__] = dict()

        for val in vals:
            preds = func(distribution, val)
            rec = recall(ground_truth, preds)
            prec = precision(ground_truth, preds)
            results[distrib_type][func.__name__][val] = {
                'precision': prec
            }
print(json.dumps(results, indent=2))
with open('results.json', 'w') as f:
    f.write(json.dumps(results, indent=2))






