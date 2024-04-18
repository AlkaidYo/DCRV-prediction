import sys
sys.path.append('../')
import matplotlib.pyplot as plt
import numpy as np


from config.file_path_config import base_path


def start_plot():
    NC_time = [0.046, 0.215, 0.355]
    VM_time = [0.728, 0.764, 0.780]
    HY_time = [0.745, 0.771, 0.783]
    MX_time = [0.748, 0.774, 0.786]
    labels = ['100', '250', '500']


    bar_width = 0.15

    r1 = np.arange(len(NC_time))
    r2 = [x + bar_width for x in r1]
    r3 = [x + bar_width for x in r2]
    r4 = [x + bar_width for x in r3]

    plt.figure(figsize=(6, 6))
    plt.bar(r1, NC_time, color='#E5DEDE', edgecolor='black', width=bar_width, label='Node', hatch='//')
    plt.bar(r2, VM_time, color='#CCBCBC', edgecolor='black', width=bar_width, label='VM')
    plt.bar(r3, HY_time, color='#9BADB2', edgecolor='black', width=bar_width, label='Hybrid', hatch='\\\\')
    plt.bar(r4, MX_time, color='#666666', edgecolor='black', width=bar_width, label='Mixture')


    plt.xlabel('Cost of node mitigation')


    plt.xticks([r + bar_width for r in range(len(NC_time))], labels)
    plt.ylim(0, 1)

    plt.xticks(fontsize=22)
    plt.yticks(fontsize=22)
    plt.legend(loc='upper right', ncol=2, mode="expand")
    plt.savefig(base_path + 'figure/figure_9b.pdf', dpi=600, bbox_inches='tight')

    plt.show()


if __name__ == '__main__':
    start_plot()
