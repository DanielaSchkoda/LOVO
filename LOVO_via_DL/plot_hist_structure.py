import matplotlib
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

if __name__ == '__main__':
    font = {'weight': 'bold', 'size': 16}
    plt.rc('font', **font)
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42

    df = pd.read_csv('./simulation_results/structure_test_res.csv', index_col=0)
    #sns.set_style("whitegrid")
    bp = sns.boxplot(data=df, showfliers=False, palette='tab10', saturation=1)
    # Get the current axes object
    ax = plt.gca()

    # Modify the font properties of the x-axis labels
    for label in ax.get_xticklabels():
        label.set_fontweight('normal')
    bp.set_yscale("log")
    #plt.xlabel('Pr')
    plt.ylabel('Cross Entropy for Causal Structure')
    #plt.legend()
    plt.tight_layout()
    plt.savefig('./simulation_results/structure_ce.eps')
    plt.show()