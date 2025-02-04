import numpy as np
import pandas as pd
from matplotlib import ticker
from scipy.spatial.distance import jensenshannon, cityblock
from constants import *
import matplotlib.pyplot as plt


def normalize_hist(hist):
    bins = hist['bins']
    counts = hist['counts']
    counts = counts / max(sum(counts),1)
    counts = counts * 100
    return np.c_[bins, counts]


def plot_explanation(explanation):

    data = []
    for c, attr, hist_all, hist_cluster in explanation:
        hist_rest = hist_all.copy()
        no_clust_label = 'Rest'
        hist_rest['counts'] = hist_rest['counts'] - hist_cluster['counts']
        hist_rest.loc[hist_rest['counts'] <= 0, 'counts'] = 0
        hist_rest = normalize_hist(hist_rest)
        hist_cluster = normalize_hist(hist_cluster)
        ylabel = 'frequency (%)'

        hist_rest = pd.DataFrame(hist_rest, columns=[attr, no_clust_label])
        hist_cluster = pd.DataFrame(hist_cluster, columns=[attr, f'Cluster {c}'])
        plot_df = pd.merge(hist_rest, hist_cluster)
        plot_df = plot_df[plot_df[no_clust_label] + plot_df[f'Cluster {c}'] > 3]
        # print(plot_df)
        data.append({
            'plot_df': plot_df,
            'no_clust_label': no_clust_label,
            # 'cluster_label': f'Cluster {c + 1}',
            'cluster_label': f'Cluster {c}',
            'ylabel': ylabel,
            'attr': attr
        })
    fig = plt.figure(figsize=(12, 2.3))
    plt.subplots_adjust(hspace=0.25, wspace=0.25)
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
        rotate = 90

        ax.set_xticklabels(plot_df[dat['attr']], rotation=rotate, ha='center')  # Rotate and align text

        ax.tick_params(axis='both', which='major', labelsize=14)  # Increase font size
        ax.tick_params(axis='x', which='major', pad=2.5, labelsize=12)
        if j == 0:
            ax.set_ylabel(dat['ylabel'], fontsize=13.5, labelpad=10)  # Increase font size
        ax.yaxis.set_major_locator(ticker.MultipleLocator(base=25))
        ax.grid(True, which='major', linestyle='--', linewidth=0.5, color='gray', axis='y')
        # ax.legend(fontsize=20)  # Increase legend font size
        ax.legend(loc='best', fontsize=11, framealpha=0.5)
        ax.yaxis.set_major_locator(ticker.MultipleLocator(base=10))
        j += 1
    # plt.tight_layout()
    plt.show()
