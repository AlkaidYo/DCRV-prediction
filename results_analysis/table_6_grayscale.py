import sys

sys.path.append('../')
import pandas as pd

from config.file_path_config import base_path
import numpy as np

start = 0.4
stop = 1.0
step = int((stop - start) / 0.02)
x_values = np.round(np.arange(0.4, 1.0, 0.02), 2)


def start_plot():
    y_mixture_df = pd.read_csv(base_path + 'temp_dir/table_6_mixture.csv', header=None)
    y_hybrid_df = pd.read_csv(base_path + 'temp_dir/table_6_hybrid.csv', header=None)

    y_mixture = y_mixture_df.iloc[0].tolist()
    y_hybrid = y_hybrid_df.iloc[0].tolist()


    max_y_func2 = max(y_hybrid)
    print('Grayscale test hybrid migration', max_y_func2)
    max_y_func1 = max(y_mixture)
    print('Grayscale test mixture migration',max_y_func1)



def compute_nc_vm_hybrid_mixture():
    from results_analysis.sub_fig_or_tab import table_6_grayscale_hybrid as hybrid
    from results_analysis.sub_fig_or_tab import table_6_grayscale_mixture as mixture
    from results_analysis.sub_fig_or_tab import table_6_grayscale_nc_vm as nc_vm
    nc_vm.start()
    hybrid.start()
    mixture.start()


if __name__ == '__main__':
    compute_nc_vm_hybrid_mixture()
    start_plot()
