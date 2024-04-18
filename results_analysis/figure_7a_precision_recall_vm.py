import sys

sys.path.append('../')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator
import pandas as pd
import argparse

from config.file_path_config import base_path

start = 0.4
stop = 0.78
step = int((stop - start) / 0.02) + 1
x_values = np.linspace(start, stop, step)


def start_polt(seed_value):
    df_vm = pd.read_csv(base_path + 'results/XGB/' + 'vm_test_result_' + 'tree_' + str(30) + '_seed_' + str(
        seed_value) + '_First.csv', header=0)

    y_values_line1 = df_vm['recall'].tolist()[:-10]
    y_values_line2 = df_vm['precision'].tolist()[:-10]
    y_values_line3 = df_vm['f1'].tolist()[:-10]

    plt.figure(figsize=(6, 6))
    plt.plot(x_values, y_values_line1, label='Recall', marker='s', color='green', linewidth=2.5)
    plt.plot(x_values, y_values_line2, label='Precision', marker='^', color='blue', linewidth=2.5)
    plt.plot(x_values, y_values_line3, label='F1', marker='o', color='red', linewidth=2.5)

    plt.xlabel('Threshold')
    plt.legend()
    ax = plt.gca()
    x_major_step = (ax.get_xticks()[1] - ax.get_xticks()[0]) / 5
    y_major_step = (ax.get_yticks()[1] - ax.get_yticks()[0]) / 5

    ax.xaxis.set_minor_locator(MultipleLocator(x_major_step))
    ax.yaxis.set_minor_locator(MultipleLocator(y_major_step))

    max_x = x_values[np.argmax(y_values_line3)]
    max_y = np.max(y_values_line3)
    plt.axvline(x=max_x, color='red', linestyle='--', linewidth=1)
    plt.text(max_x, max_y / 3, 'Best F1', color='red', ha='right', va='center')

    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    ax.minorticks_on()
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.grid(axis='x', linestyle='--', alpha=0.5)

    plt.savefig(base_path + 'figure/figure_7a.pdf', dpi=600, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Figure 7a: the threshold selection.')
    parser.add_argument('--seed', default='80',
                        help='the value of path/to/results/XGB/vm_test_result_tree_30_seed_????_First.csv at the position of the question mark(Default value is 80)')
    args = parser.parse_args()
    seed_value = args.seed
    start_polt(seed_value)
