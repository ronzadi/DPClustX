import itertools
import math

# import numpy as np
# import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import utils
from utils import *
from diffprivlib.mechanisms import GeometricTruncated
from sys import maxsize
import warnings

from one_shot_topk import noisy_topk


class ClusteringExplainer:
    def __init__(self, dataset, domains=None, clusters_ids=None, interest_weight=1/3, suff_weight=1/3, diversity_weight=1/3,
                 cluster_col_name='cluster', random_state=None, num_candidates=3, scores_cache=None, mode=None):

        self.dataset = dataset.copy()
        self.random_state = random_state
        self.interest_weight = interest_weight / (interest_weight + suff_weight + diversity_weight)
        self.suff_weight = suff_weight / (interest_weight + suff_weight + diversity_weight)
        self.diversity_weight = diversity_weight / (interest_weight + suff_weight + diversity_weight)
        self.cluster_col_name = cluster_col_name
        self.domains = self.get_domains(domains)
        self.attributes = self.get_attributes(dataset)
        self.clusters = self.dataset[cluster_col_name]
        self.clusters_ids = self.get_clusters_ids(clusters_ids)
        self.num_clusters = len(self.clusters_ids)
        self.num_candidates = num_candidates
        self.global_explanation = None
        self._scores_cache = scores_cache or self.init_cache()
        self._global_score = None
        self.top_comb = None
        self.cluster_sizes = self.get_cluster_sizes()
        self.dataset_histograms = None
        self.cluster_histograms = None
        self.get_dataset_histograms()
        self.get_cluster_histograms()
        self.mode = mode

    def get_cluster_sizes(self):
        return self.dataset[self.cluster_col_name].value_counts().to_dict()

    def get_cluster_histograms(self):
        self.cluster_histograms = dict()
        for cluster_id in self.clusters_ids:
            self.cluster_histograms[cluster_id] = dict()
            cluster = self.dataset[self.clusters == cluster_id]
            for a in self.attributes:
                self.cluster_histograms[cluster_id][a] = self.generate_single_hist(self.domains[a], cluster[a],
                                                                                   1, False)

    def get_dataset_histograms(self):

        self.dataset_histograms = dict()
        for a in self.attributes:
            self.dataset_histograms[a] = self.generate_single_hist(self.domains[a], self.dataset[a], 1,
                                                                   False)

    def init_cache(self):
        scores = {
            SUFF: {True: dict(), False: dict()},
            INTEREST: {True: dict(), False: dict()},
            DIVERS: {True: dict(), False: dict()},
            SINGLE: {True: dict(), False: dict()},
        }
        return scores

    def get_domains(self, domains):
        if domains is None:
            # warnings.warn("Domains not provided. Using domains from input dataset.")
            domains = {
                col: np.unique(self.dataset[col])
                for col in self.dataset.columns
            }
        return domains

    def get_clusters_ids(self, cluster_ids):
        if cluster_ids is None:
            # warnings.warn("Cluster ids not provided. Using cluster ids from input dataset.")
            cluster_ids = np.unique(self.clusters)
        return cluster_ids


    def sufficiency(self, attribute, cluster_id, normalize=False):

        counts_d = self.dataset_histograms[attribute]
        counts_c = self.cluster_histograms[cluster_id][attribute]
        res = pd.merge(counts_d, counts_c, how='inner', on='bins', suffixes=('_d', '_c'))
        res['counts_d'] = res[['counts_d', 'counts_c']].max(axis=1)
        res = res[res['counts_d'] > 0]
        res['score'] = res['counts_c'] ** 2 / res['counts_d']
        suff = res['score'].sum()
        if normalize:
            suff = suff / res['counts_c'].sum()
        return suff

    def interest(self, attribute, cluster_id, normalize=False):

        counts_d = self.dataset_histograms[attribute]
        counts_c = self.cluster_histograms[cluster_id][attribute]
        res = pd.merge(counts_d, counts_c, how='left', on='bins', suffixes=('_d', '_c')).fillna(0)

        cluster_size = counts_c['counts'].sum()
        dataset_size = counts_d['counts'].sum()

        res['score'] = abs(res['counts_c'] / cluster_size - res['counts_d'] / dataset_size)
        inter = 0.5 * res['score'].sum()
        if not normalize:
            inter = inter * cluster_size
        return inter

    def pair_diversity(self,cluster_1, cluster_2, attr_1, attr_2, normalize=False):

        pair = (cluster_1, cluster_2, attr_1, attr_2)
        sym_pair = (cluster_2, cluster_1, attr_2, attr_1)
        if pair in self._scores_cache[DIVERS][normalize]:
            return self._scores_cache[DIVERS][normalize][pair]

        elif sym_pair in self._scores_cache[DIVERS][normalize]:
            return self._scores_cache[DIVERS][normalize][sym_pair]

        s_1 = (self.clusters == cluster_1).sum()
        s_2 = (self.clusters == cluster_2).sum()
        minsize = 1 if normalize else min(s_1, s_2)
        if attr_1 != attr_2:
            return minsize

        counts_1 = self.cluster_histograms[cluster_1][attr_1]
        counts_2 = self.cluster_histograms[cluster_2][attr_2]
        res = pd.merge(counts_1, counts_2, how='outer', on='bins', suffixes=('_1', '_2')).fillna(0)
        res['score'] = abs(res['counts_1'] / max(s_1, 1) - res['counts_2'] / max(s_2,1))
        diversity = minsize * 0.5 * res['score'].sum()

        self._scores_cache[DIVERS][normalize][sym_pair] = self._scores_cache[DIVERS][normalize][
            pair] = diversity

        return diversity


    def get_attr_diversity(self, attr, attr_combination, normalize):

        attr_diversity = []
        exp_by_attr = [i for i, col in enumerate(attr_combination) if col == attr]

        for i, j in itertools.combinations(range(len(exp_by_attr)), 2):
            if i >= j:
                continue
            attr_1 = attr_combination[exp_by_attr[i]]
            attr_2 = attr_combination[exp_by_attr[j]]
            cluster_1 = self.clusters_ids[exp_by_attr[i]]
            cluster_2 = self.clusters_ids[exp_by_attr[j]]

            pair = (cluster_1, cluster_2, attr_1, attr_2)
            sym_pair = (cluster_2, cluster_1, attr_2, attr_1)
            if pair in self._scores_cache[DIVERS][normalize]:
                pair_diversity = self._scores_cache[DIVERS][normalize][pair]

            elif sym_pair in self._scores_cache[DIVERS][normalize]:
                pair_diversity = self._scores_cache[DIVERS][normalize][sym_pair]

            else:
                pair_diversity = self.pair_diversity(cluster_1, cluster_2, attr_1, attr_2, normalize)
                self._scores_cache[DIVERS][normalize][sym_pair] = self._scores_cache[DIVERS][normalize][
                    pair] = pair_diversity

            attr_diversity.append(pair_diversity)

        return np.mean(attr_diversity)

    def global_diversity(self, attr_combination, normalize=False):

        score = 0
        for i, j in itertools.combinations(range(len(attr_combination)), 2):
            attr_1 = attr_combination[i]
            attr_2 = attr_combination[j]
            cluster_1 = self.clusters_ids[i]
            cluster_2 = self.clusters_ids[j]

            pair = (cluster_1, cluster_2, attr_1, attr_2)
            sym_pair = (cluster_2, cluster_1, attr_2, attr_1)
            if pair in self._scores_cache[DIVERS][normalize]:
                pair_diversity = self._scores_cache[DIVERS][normalize][pair]

            elif sym_pair in self._scores_cache[DIVERS][normalize]:
                pair_diversity = self._scores_cache[DIVERS][normalize][sym_pair]

            else:
                pair_diversity = self.pair_diversity(cluster_1, cluster_2, attr_1, attr_2, normalize)
                self._scores_cache[DIVERS][normalize][sym_pair] = self._scores_cache[DIVERS][normalize][
                    pair] = pair_diversity

            score += pair_diversity

        score = score / (len(attr_combination) * (len(attr_combination) - 1) / 2)

        return score

    def score_single(self, cluster_id, attribute, normalize=False):

        # if (cluster_id, attribute) in self._scores_cache[SINGLE]:
        #     return self._scores_cache[SINGLE][(cluster_id, attribute)]

        lambda_int = self.interest_weight / (self.interest_weight + self.suff_weight)
        lambda_suff = self.suff_weight / (self.interest_weight + self.suff_weight)

        score = (self.interest(attribute, cluster_id, normalize) * lambda_int +
                 self.sufficiency(attribute, cluster_id, normalize) * lambda_suff)

        # self._scores_cache[SINGLE][(cluster_id, attribute)] = score

        if score == 0:
            print('zero score', cluster_id, attribute)

        return score

    def global_interest(self, attr_combination, normalize=False):
        score = 0
        for i, attr in enumerate(attr_combination):
            cluster_id = self.clusters_ids[i]
            if (cluster_id, attr) not in self._scores_cache[INTEREST][normalize]:
                self._scores_cache[INTEREST][normalize][(cluster_id, attr)] = self.interest(attr, cluster_id, normalize)
            score += self._scores_cache[INTEREST][normalize][(cluster_id, attr)]
        score = score / (len(attr_combination))
        return score

    def global_suff(self, attr_combination, normalize=False):
        score = 0
        for i, attr in enumerate(attr_combination):
            cluster_id = self.clusters_ids[i]
            if (cluster_id, attr) not in self._scores_cache[SUFF][normalize]:
                self._scores_cache[SUFF][normalize][(cluster_id, attr)] = self.sufficiency(attr, cluster_id, normalize)
            score += self._scores_cache[SUFF][normalize][(cluster_id, attr)]
        score = score / (len(attr_combination))
        return score

    def score_global(self, attr_combination, normalize=False):

        score = 0

        diversity_g = self.global_diversity(attr_combination, normalize)
        suff_g = self.global_suff(attr_combination, normalize)
        interest_g = self.global_interest(attr_combination, normalize)

        score += self.diversity_weight * diversity_g
        score += self.suff_weight * suff_g
        score += self.interest_weight * interest_g
        return score, interest_g, suff_g, diversity_g

    def top_single_cluster_exp_attrs(self, cluster_id, eps_single, k, private=True):

        scores = {A: self.score_single(cluster_id, A) for A in self.attributes}
        topk = noisy_topk(scores, k, eps_single, 1, private)

        return topk

    def get_attributes(self, dataset):
        # Single columns
        attributes = dataset.columns.tolist()
        attributes.remove(self.cluster_col_name)
        return attributes

    def generate_single_hist(self, bins, arr, eps, private=True):

        # eps = math.sqrt(2 * eps)
        eps = eps

        hist = arr.value_counts().reindex(bins).fillna(0).reset_index()
        counts = hist.iloc[:, 1]

        if not private:
            return pd.DataFrame(data=np.c_[bins, counts], columns=['bins', 'counts'])

        dp_mech = GeometricTruncated(
            epsilon=eps,
            sensitivity=1,
            lower=0,
            upper=maxsize,
            random_state=self.random_state,
        )
        dp_counts = np.zeros_like(counts)

        for i, count in enumerate(counts):
            dp_counts[i] = dp_mech.randomise(int(count))

        df = pd.DataFrame(data=np.c_[bins, dp_counts], columns=['bins', 'counts'])
        return df

    def gen_noisy_hists(self, attributes, eps_hist, private=True):

        attr_set = set(attributes)
        eps_hist_all = eps_hist / (2 * len(attr_set))
        full_hists = {
            a: self.generate_single_hist(self.domains[a], self.dataset[a], eps_hist_all, private)
            for a in attr_set
        }

        eps_hist_cluster = eps_hist / 2
        cluster_hists = []
        for i, a in enumerate(attributes):
            cluster_id = self.clusters_ids[i]
            cluster = self.dataset[self.clusters == cluster_id]
            cluster_hists.append(self.generate_single_hist(self.domains[a], cluster[a], eps_hist_cluster, private))

        global_explanation = []
        for i, cluster_hist in enumerate(cluster_hists):
            single_exp = [
                self.clusters_ids[i], attributes[i], full_hists[attributes[i]], cluster_hist
            ]
            global_explanation.append(single_exp)

        return global_explanation

    def get_budgets(self, eps_candlist, eps_topcomb, eps_hist):
        if self.mode == USER:
            return eps_candlist or 1/30, eps_topcomb or 1/30, eps_hist or 1/30

    def explain_admin(self, eps_candlist, eps_topcomb, eps_hist):

        priv_candlist = eps_candlist is not None
        priv_topcomb = eps_topcomb is not None
        priv_hist = eps_hist is not None

        explanation_candidates = []

        # Get top explaining attributes for each cluster
        eps_single = eps_candlist / self.num_clusters
        for cluster_id in self.clusters_ids:
            single_exp = self.top_single_cluster_exp_attrs(
                cluster_id, eps_single, self.num_candidates, priv_candlist
            )
            # print(cluster_id, single_exp)
            explanation_candidates.append(single_exp)

        # Select noisy-best combination
        scores = {
            comb: self.score_global(comb)[0]
            for comb in itertools.product(*explanation_candidates)
        }

        if self.diversity_weight > 0:
            top_comb = noisy_topk(scores, 1, eps_topcomb, 1, priv_topcomb)[0]
        else:
            assert len(scores.keys()) == 1
            top_comb = next(iter(scores))

        self._global_score = scores[top_comb]
        self.top_comb = top_comb
        # print(top_comb)
        # Generate noisy histograms
        global_explanation = self.gen_noisy_hists(top_comb, eps_hist, priv_hist)
        self.global_explanation = global_explanation
        self.show_explanation()

        return global_explanation

    def explain_user(self, eps_candlist, eps_topcomb, eps_hist):

        eps_candlist = eps_candlist or DEFAULT_BUDGET
        eps_topcomb = eps_topcomb or DEFAULT_BUDGET
        eps_hist = eps_hist or DEFAULT_BUDGET

        explanation_candidates = []

        # Get top explaining attributes for each cluster
        eps_single = eps_candlist / self.num_clusters
        for cluster_id in self.clusters_ids:
            single_exp = self.top_single_cluster_exp_attrs(
                cluster_id, eps_single, self.num_candidates, private=True
            )
            # print(cluster_id, single_exp)
            explanation_candidates.append(single_exp)

        # Select noisy-best combination
        scores = {
            comb: self.score_global(comb)[0]
            for comb in itertools.product(*explanation_candidates)
        }

        if self.diversity_weight > 0:
            top_comb = noisy_topk(scores, 1, eps_topcomb, 1, private=True)[0]
        else:
            assert len(scores.keys()) == 1
            top_comb = next(iter(scores))

        self._global_score = scores[top_comb]
        self.top_comb = top_comb
        # print(top_comb)
        # Generate noisy histograms
        global_explanation = self.gen_noisy_hists(top_comb, eps_hist, private=True)
        self.global_explanation = global_explanation
        self.show_explanation()

        return global_explanation

    def explain(self, eps_candlist=None, eps_topcomb=None, eps_hist=None):

        if self.mode == USER:
            self.explain_user(eps_candlist, eps_topcomb, eps_hist)
        elif self.mode == ADMIN:
            self.explain_admin(eps_candlist, eps_topcomb, eps_hist)

    def show_explanation(self):
        utils.plot_explanation(self.global_explanation)

    def explain1(self, eps_hist=0.1,  private=True):

        text_desctiptions1 = {
            1: " $\\mathbf{'Age'}$ values below $49$ are $13$ times more frequent in Cluster 1 "
               "($76\\%$) than in the remaining data ($6\\%$).",
            2: " $\\mathbf{'Age'}$ values above $60$ are $7$ times more frequent in Cluster 2 "
               "($82\\%$) than in the remaining data ($12\\%$).",
            3: " $\\mathbf{'Income'}$ values below $25K$ are $8$ times more frequent in Cluster 3 "
               "($85\\%$) than in the remaining data ($11\\%$)."
        }

        data = []

        explanation = self.gen_noisy_hists(('Age', 'Age', 'Income'), eps_hist, private)
        # Generate data and store it in the list
        for c, attr, hist_all, hist_cluster in explanation:
            no_clust_label = 'Dataset'
            ylabel = 'Count'
            hist_rest = hist_all.copy()
            if True:
                no_clust_label = 'Rest'
                hist_rest['counts'] = hist_rest['counts'] - hist_cluster['counts']
                hist_rest.loc[hist_rest['counts'] <= 0, 'counts'] = 0
            if True:
                hist_rest = normalize_hist(hist_rest)
                hist_cluster = normalize_hist(hist_cluster)
                ylabel = 'frequency (%)'

            hist_rest = pd.DataFrame(hist_rest, columns=[attr, no_clust_label])
            hist_cluster = pd.DataFrame(hist_cluster, columns=[attr, f'Cluster {c + 1}'])
            plot_df = pd.merge(hist_rest, hist_cluster)
            plot_df = plot_df[plot_df[no_clust_label] + plot_df[f'Cluster {c + 1}'] > 3]
            # print(plot_df)
            # Store the data and labels
            data.append({
                'plot_df': plot_df,
                'no_clust_label': no_clust_label,
                'cluster_label': f'Cluster {c + 1}',
                'ylabel': ylabel,
                'attr': attr
            })
        # fig = plt.figure(figsize=(20, 6.2)) # For opt
        fig = plt.figure(figsize=(12, 2.3))  # For DP
        plt.subplots_adjust(hspace=0.25, wspace=0.25)
        # create 3x1 subfigs

        axs = fig.subplots(nrows=1, ncols=len(explanation))

        axs = axs.flatten()
        j = 0
        for ax, dat in zip(axs, data):
            plot_df = dat['plot_df']
            bar_width = 0.43  # Adjust bar width
            r1 = np.arange(len(plot_df[dat['attr']]))
            r2 = [x + bar_width for x in r1]

            ax.bar(r2, plot_df[dat['cluster_label']], width=bar_width, edgecolor='grey', label=dat['cluster_label'],
                   color='#5573CD')
            ax.bar(r1, plot_df[dat['no_clust_label']], width=bar_width, edgecolor='grey', label=dat['no_clust_label'],
                   color='#CD5573')

            ax.set_xlabel(f"'{dat['attr']}'", fontsize=13.5, labelpad=10)  # Increase font size
            ax.set_xticks([r + bar_width / 2 for r in r1])
            if dat['attr'] == 'GenHlth':
                rotate = 45
            elif dat['attr'] == 'Age':
                rotate = 90
            else:
                rotate = 90

            ax.set_xticklabels(plot_df[dat['attr']], rotation=rotate, ha='center')  # Rotate and align text

            if dat['attr'] == 'GenHlth':
                ax.set_xticklabels(['Excellent', 'Very good', 'Good', 'Fair', 'Poor'])

            ax.tick_params(axis='both', which='major', labelsize=14)  # Increase font size
            ax.tick_params(axis='x', which='major', pad=2.5, labelsize=12)
            if j == 0:
                ax.set_ylabel(dat['ylabel'], fontsize=13.5, labelpad=10)  # Increase font size
            ax.yaxis.set_major_locator(ticker.MultipleLocator(base=25))
            ax.grid(True, which='major', linestyle='--', linewidth=0.5, color='gray', axis='y')
            # ax.legend(fontsize=20)  # Increase legend font size

            if j == 2:
                ax.legend(loc='best', fontsize=11, framealpha=0.5)
                ax.yaxis.set_major_locator(ticker.MultipleLocator(base=10))
                # ax.legend(loc='best',fontsize=8, framealpha=0.5)
                # ax.legend(fontsize=22)
            else:
                ax.legend(loc='upper left', fontsize=11, framealpha=0.5)
                # ax.legend(fontsize=26, framealpha=0.7)
            if dat['attr'] == 'Age':
                ax.yaxis.set_major_locator(ticker.MultipleLocator(base=10))

            wrapped = wrap_text(text_desctiptions1[j + 1], 31)
            ax.text(
                0.5, 1.35, wrapped,  # x, y position (relative to the axes)
                transform=ax.transAxes,  # Use axis coordinates for positioning
                fontsize=12,
                ha='center',  # Horizontal alignment
                va='center',  # Vertical alignment
                # bbox=dict(boxstyle="round,pad=0.3", edgecolor='black', facecolor='white')  # Bounding box style
            )
            j += 1
        # plt.tight_layout()
        plt.show()






















