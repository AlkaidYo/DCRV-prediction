import sys

sys.path.append('../')
import pandas as pd

from config.file_path_config import base_path

sys.path.append('/mnt3/sc')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator, NullFormatter
import matplotlib.ticker as ticker
import argparse

start = 0.4
stop = 1.0
step = int((stop - start) / 0.02)
x_values = np.round(np.arange(0.4, 1.0, 0.02), 2)


def start_plot():
    y_mixture_df = pd.read_csv(base_path + 'temp_dir/figure_8b_mixture.csv', header=None)
    y_hybrid_df = pd.read_csv(base_path + 'temp_dir/figure_8b_hybrid.csv', header=None)

    y_mixture = y_mixture_df.iloc[0].tolist()
    y_hybrid = y_hybrid_df.iloc[0].tolist()

    plt.figure(figsize=(6, 6))
    plt.plot(x_values, y_mixture, marker='^', label='Mixture', linewidth=2.5)
    plt.plot(x_values, y_hybrid, marker='D', label='Hybrid', linewidth=2.5)
    plt.xlabel('Threshold')

    plt.legend(loc='lower right')

    ax = plt.gca()
    x_major_step = (ax.get_xticks()[1] - ax.get_xticks()[0]) / 5
    y_major_step = (ax.get_yticks()[1] - ax.get_yticks()[0]) / 5

    ax.xaxis.set_minor_locator(MultipleLocator(x_major_step))
    ax.set_ylim(0.4, 0.85)
    ax.yaxis.set_major_locator(MultipleLocator(0.1))
    ax.yaxis.set_minor_locator(MultipleLocator(0.02))
    ax.yaxis.set_minor_formatter(NullFormatter())


    max_y_func1 = max(y_mixture)
    max_x_func1 = x_values[y_mixture.index(max_y_func1)]
    ax.annotate('Max:{:.3f}'.format(max_y_func1),
                xy=(max_x_func1, max_y_func1),
                xytext=(max_x_func1, max_y_func1 + 0.053),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3', color='black', lw=1),
                verticalalignment='top',
                horizontalalignment='center')


    max_y_func2 = max(y_hybrid)
    max_x_func2 = x_values[y_hybrid.index(max_y_func2)]
    ax.annotate('Max:{:.3f}'.format(max_y_func2),
                xy=(max_x_func2, max_y_func2),
                xytext=(max_x_func2, max_y_func2 - 0.04),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3', color='black', lw=1),
                verticalalignment='top',
                horizontalalignment='center')
    plt.xticks(fontsize=22)
    plt.yticks(fontsize=22)
    plt.gca().yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.grid(axis='x', linestyle='--', alpha=0.5)
    plt.savefig(base_path + 'figure/figure_8b.pdf', dpi=600, bbox_inches='tight')
    plt.show()


def compute_nc_vm_hybrid_mixture(seed_value=80):
    from results_analysis.sub_fig_or_tab import figure_8b_hybrid as hybrid
    from results_analysis.sub_fig_or_tab import figure_8b_mixture as mixture
    hybrid.start(seed_value)
    mixture.start(seed_value)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Figure 8b')
    parser.add_argument('--seed', default='80',
                        help='the value of path/to/results/XGB/vm_test_result_tree_30_seed_????_First.csv at the position of the question mark(Default value is 80)')
    args = parser.parse_args()
    seed_value = args.seed
    compute_nc_vm_hybrid_mixture(seed_value)
    start_plot()
