import matplotlib
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import numpy as np

if __name__ == '__main__':
    font = {'weight': 'bold', 'size': 16}
    plt.rc('font', **font)
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42
    plt.rcParams['axes.titleweight'] = 'bold'

    df = pd.read_csv('./simulation_results/pred_error_res.csv', index_col=0)
    sns.set_palette('tab10')
    dotted_line = np.linspace(0, max(df['error_dl']), 100)
    plt.plot(dotted_line, dotted_line, color='black', zorder=-1)
    plt.scatter(df['error_base'], df['error_dl'], alpha=.8)
    plt.xlabel('Baseline loss')
    plt.ylabel('Deep Learning LOVO loss')
    plt.title('Deep Learning LOVO', pad=10)
    plt.legend()
    plt.tight_layout()
    plt.savefig('./simulation_results/DL_error.eps')
    plt.show()