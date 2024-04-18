import sys

sys.path.append('../')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import argparse
from config.file_path_config import base_path

def start_plot():
    nc_vm_df = pd.read_csv(base_path + 'temp_dir/figure_7b_nc_vm.csv', header=None)
    mix_df = pd.read_csv(base_path + 'temp_dir/figure_7b_mixture.csv', header=None)
    hyb_df = pd.read_csv(base_path + 'temp_dir/figure_7b_hybrid.csv', header=None)
    NC_time = nc_vm_df.iloc[1].tolist()
    NC_time = [x / 60 for x in NC_time]
    VM_time = nc_vm_df.iloc[0].tolist()
    VM_time = [x / 60 for x in VM_time]
    HY_time = hyb_df.iloc[0].tolist()
    HY_time = [x / 60 for x in HY_time]
    MX_time = mix_df.iloc[0].tolist()
    MX_time = [x / 60 for x in MX_time]


    labels = ['5', '50', '500']


    bar_width = 0.15


    r1 = np.arange(len(NC_time))
    r2 = [x + bar_width for x in r1]
    r3 = [x + bar_width for x in r2]
    r4 = [x + bar_width for x in r3]


    plt.figure(figsize=(6, 6))
    plt.bar(r1, NC_time, color='#E5DEDE', edgecolor='black', width=bar_width, label='Node')
    plt.bar(r2, VM_time, color='#CCBCBC', edgecolor='black', width=bar_width, label='VM')
    plt.bar(r3, HY_time, color='#9BADB2', edgecolor='black', width=bar_width, label='Hybrid')
    plt.bar(r4, MX_time, color='#666666', edgecolor='black', width=bar_width, label='Mixture')


    plt.xlabel('Node Crash Time(h)')


    plt.yticks([0, 10000, 20000, 30000, 40000, 50000], ['0', '1e4', '2e4', '3e4', '4e4', '5e4'])

    plt.xticks([r + bar_width for r in range(len(NC_time))], labels)
    plt.xticks(fontsize=22)
    plt.yticks(fontsize=22)

    plt.legend()
    plt.savefig(base_path + 'figure/figure_7b.pdf', dpi=600, bbox_inches='tight')

    plt.show()


def compute_nc_vm_hybrid_mixture(seed_value=80):
    from results_analysis.sub_fig_or_tab import figure_7b_loss_time_nc_vm as ncvm
    from results_analysis.sub_fig_or_tab import figure_7b_loss_time_hybrid as hybrid
    from results_analysis.sub_fig_or_tab import figure_7b_loss_time_mixture as mixture
    ncvm.start(seed_value)
    hybrid.start(seed_value)
    mixture.start(seed_value)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Figure 7b: loss compute time.')
    parser.add_argument('--seed', default='80',
                        help='the value of path/to/results/XGB/vm_test_result_tree_30_seed_????_First.csv at the position of the question mark(Default value is 80)')
    args = parser.parse_args()
    seed_value = args.seed
    compute_nc_vm_hybrid_mixture(seed_value)
    start_plot()
