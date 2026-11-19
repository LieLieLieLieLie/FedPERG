"""Paired seed-cluster uncertainty for experiment summaries."""
import itertools

import numpy as np


def interval(values):
    values = np.asarray(values, dtype=float)
    count = len(values)
    if not count:
        return None
    generator = np.random.default_rng(93297)
    draws = values[generator.integers(count, size=(20000, count))].mean(1)
    low, high = np.quantile(draws, [.025, .975])
    probability = None
    if count <= 16:
        signs = np.array(list(itertools.product([-1, 1], repeat=count)))
        probability = float((np.abs((signs * values).mean(1)) >= abs(values.mean()) - 1e-12).mean())
    return {
        'mean': float(values.mean()), 'ci95': [float(low), float(high)], 'n': count,
        'exact_two_sided_p': probability, 'positive': int((values > 0).sum()),
        'negative': int((values < 0).sum()), 'values': values.tolist(),
    }
