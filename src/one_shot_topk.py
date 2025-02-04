from typing import Any, Dict
import heapq
import numpy as np


def noisy_topk(candidates_with_scores: Dict[Any, float], k, eps, sensitivity, private=True):
    # bounded range DP, which implies 1/8 \eps^2-zCDP

    if private:
        epsilon = eps / k
        gumbel_scores = {
            candidate: score + gumbel_noise(2 * sensitivity / epsilon)
            for candidate, score in candidates_with_scores.items()
        }
    else:
        gumbel_scores = candidates_with_scores

    topk = heapq.nlargest(k, gumbel_scores.keys(), key=gumbel_scores.get)

    return topk


def gumbel_noise(scale):
    return np.random.gumbel(scale=scale)