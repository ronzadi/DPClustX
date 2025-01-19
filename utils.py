import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon, cityblock
from constants import *

def calculate_distance(s, r, data_type=None):
    s_np = np.array(s, dtype='float64')
    s_np = s_np[s_np == s_np]
    r_np = np.array(r, dtype='float64')
    r_np = r_np[r_np == r_np]
    # if data_type == NUMERIC_TYPE:
    #     return 0 if len(r) == 0 else wasserstein_distance(r, s)
    # else:
    try:
        return 0 if len(r_np) == 0 else jensenshannon(r_np, s_np)
    except Exception:
        return
def calculate_tvd(s, r, data_type=None):
    # s_np = np.array(s, dtype='float64')
    # s_np = s_np[s_np == s_np]
    # r_np = np.array(r, dtype='float64')
    # r_np = r_np[r_np == r_np]

    return 0.5 * cityblock(s, r)


def distributions_distance(source_dist, result_dist, data_type=None):
    distributions_pd = pd.concat([source_dist, result_dist], axis=1).fillna(0.0)
    dist = calculate_distance(distributions_pd[source_dist.name], distributions_pd[result_dist.name], data_type)
    return dist, distributions_pd[result_dist.name].fillna(0.0)

def distributions_distance_tvd(source_dist, result_dist, data_type=None):
    distributions_pd = pd.concat([source_dist, result_dist], axis=1).fillna(0.0)
    dist = calculate_tvd(distributions_pd[source_dist.name], distributions_pd[result_dist.name], data_type)
    return dist, distributions_pd[result_dist.name].fillna(0.0)

def normalize_by_clusters_num(raw_score, num_clusters):
    return (raw_score - 1.0 / num_clusters) / (1.0 - 1.0 / num_clusters)


def determine_column_type(column_series):
    if (column_series.dtype == 'string' or column_series.dtype == 'object') or (
            len(column_series.drop_duplicates()) < 20):
        return CATEGORICAL_TYPE
    return NUMERIC_TYPE


def equalObs(x, nbin):
    nlen = len(x)
    interp = np.interp(np.linspace(0, nlen, nbin + 1), np.arange(nlen), np.sort(x))
    return np.unique(interp)


def distributions_lift(source_dist, result_dist):
    # print(f"source:\\n {source_dist}")
    # print(f"result:\\n {result_dist}")
    distributions_pd = pd.concat([source_dist, result_dist], axis=1).fillna(0.0)
    return distributions_pd[result_dist.name] / distributions_pd[source_dist.name]


def diversity_from_distributions(distributions_list):
    diversity = 1
    for i in range(1, len(distributions_list), 1):
        min_distance = np.inf
        for j in range(i):
            distance = calculate_distance(distributions_list[i], distributions_list[j])
            min_distance = distance if distance < min_distance else min_distance
        diversity += min_distance
    return diversity

def diversity_from_distributions_tvd(distributions_list):
    diversity = 1
    for i in range(1, len(distributions_list), 1):
        min_distance = np.inf
        for j in range(i):
            distance = calculate_tvd(distributions_list[i][:,1], distributions_list[j][:,1])
            min_distance = distance if distance < min_distance else min_distance
        diversity += min_distance
    return diversity




def bin_single_column(binning_column, type, data_type, num_bins=10, bins=None):
    # data_type = self.source_distributions[binning_column.name][DATA_TYPE]
    # type must be one of 'source', 'result'
    if data_type == CATEGORICAL_TYPE:
        histogram = binning_column.value_counts(normalize=True).rename(f"{binning_column.name}_{type}")
        bins = list(histogram.index)
        return histogram, bins
    elif data_type == NUMERIC_TYPE:
        if bins is None and num_bins is not None:
            bins = equalObs(binning_column, num_bins)
        counts, bin_edges = np.histogram(binning_column, bins=bins)
        counts = counts / sum(counts)
        bin_edges[len(bin_edges) - 1] = bin_edges[len(bin_edges) - 1] + 1  # patch to solve pd.cut problem last bin
        return pd.Series(counts, index=bin_edges[:-1], name=f"{binning_column.name}_{type}"), bins
    else:
        raise Exception("incorrect data type")


def best_from_list_by_order_type(lst, ordering_type):
    assert ordering_type in ['best', 'worst', 'median']
    if ordering_type == 'best':
        return np.argmax(lst)
    elif ordering_type == 'median':
        return np.argsort(lst)[len(lst) // 2]
    elif ordering_type == 'worst':
        return np.argsort(lst)[0]

def normalize_hist(hist):
    bins = hist['bins']
    counts = hist['counts']
    counts = counts / max(sum(counts),1)
    counts = counts * 100
    return np.c_[bins, counts]


def evaluate_sufficiency(hist_cohort, hist_all):
    hist_cohort = pd.DataFrame(hist_cohort, columns=['bins', 'count'])
    hist_all = pd.DataFrame(hist_all, columns=['bins', 'count'])
    res = pd.merge(hist_all, hist_cohort, how='inner', on='bins', suffixes=('_d', '_c'))
    res['score'] = res['count_c'] ** 2 / res['count_d']
    suff = (res['score'].sum()) / (hist_all['count'].sum())
    return suff

def evaluate_sufficiency(hist_cohort, hist_all):
    hist_cohort = pd.DataFrame(hist_cohort, columns=['bins', 'count'])
    hist_all = pd.DataFrame(hist_all, columns=['bins', 'count'])
    res = pd.merge(hist_all, hist_cohort, how='outer', on='bins', suffixes=('_d', '_c')).fillna(0.0)

    res['count_c'] = res[['count_d','count_c']].min(axis=1) # Make sure no infs

    res['score'] = res['count_c'] ** 2 / res['count_d']
    score = res['score'].sum()
    cnt = hist_cohort['count'].sum()
    if cnt == 0.0 or np.isnan(cnt):
        return 0.0
    return score / cnt

def evaluate_sufficiency2(hist_cohort, hist_all):
    hist_cohort = pd.DataFrame(hist_cohort, columns=['bins', 'count'])
    hist_all = pd.DataFrame(hist_all, columns=['bins', 'count'])
    res = pd.merge(hist_all, hist_cohort, how='inner', on='bins', suffixes=('_d', '_c'))
    sum_c = (res['count_c'] * res['count_d']).sum()
    sum_d = (res['count_d'] ** 2).sum()

    return sum_c / sum_d

def evaluate_sufficiency3(explained_col, cluster_id, clustered_dataset, hist_cohort, hist_all):
    hist_cohort = pd.DataFrame(normalize_hist(hist_cohort), columns=['bins', 'count'])
    hist_all = pd.DataFrame(normalize_hist(hist_all), columns=['bins', 'count'])
    res = pd.merge(hist_all, hist_cohort, how='outer', on='bins', suffixes=('_d', '_c')).fillna(0.0)
    res[PROBA_COL] = res['count_c'] / res['count_d']
    explained_lift = res[['bins', PROBA_COL]]
    binned_explained_col = explained_col + BINNED_SUFFIX
    clustered_dataset.loc[:, binned_explained_col] = clustered_dataset[explained_col]
    binned_proba_df = explained_lift.rename(
      columns={'bins': binned_explained_col})
    clustered_dataset_with_proba = clustered_dataset.merge(binned_proba_df, on=binned_explained_col, how='left')
    clustered_dataset.drop(columns=[binned_explained_col], inplace=True)
    full_dataset_proba_sum = clustered_dataset_with_proba[PROBA_COL].sum()
    cluster_proba_sum = clustered_dataset_with_proba[clustered_dataset_with_proba[CLUSTER] == cluster_id][
      PROBA_COL].sum()
    if cluster_proba_sum == 0:
        return 0.0
    return cluster_proba_sum / full_dataset_proba_sum


def evaluate_interest(hist_cohort, hist_all):
    hist_cohort = pd.DataFrame(hist_cohort, columns=['bins', 'count'])
    hist_all = pd.DataFrame(hist_all, columns=['bins', 'count'])
    res = pd.merge(hist_all, hist_cohort, how='left', on='bins', suffixes=('_d', '_c')).fillna(0)
    cohort_size = (hist_cohort['count']).sum()
    dataset_size = (hist_all['count']).sum()

    res['score'] = abs(res['count_c'] / cohort_size - res['count_d'] / dataset_size)
    inter = 0.5 * res['score'].sum()
    return inter



