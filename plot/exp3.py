from matplotlib import pyplot as plt
import numpy as np
import itertools
def plot(attack_methods, survey_methods):
    plt.switch_backend('agg')
    # xLabel = ['MV', 'WMV', 'ZC', 'DS','CATD','GLAD','KOS','VI-BP']
    # yLabel = ['MV', 'WMV', 'ZC', 'DS']
    xLabel = survey_methods
    yLabel = attack_methods
    data = {}

    base_folder = './output/exp3/'
    POAC = []
    datasets = ['temp', 'rte', 'sentiment']
    for dataset in datasets:
        temp1 = []
        with open(base_folder + "change_rate_" + dataset + "_POAC") as f:
            text = f.read().split(',')
            for i in range(len(yLabel)):
                temp2 = []
                for j in range(len(xLabel)):
                    temp2.append(float(text[i * len(xLabel) + j]))
                temp1.append(temp2)
            data[dataset] = temp1
            f.close()

    cmap = plt.get_cmap('tab20c') ###  RdGy cool tab20b tab20c

    for dataset in datasets:
        plt.figure()
        plt.imshow(data[dataset], interpolation='nearest', cmap=cmap)
        cb = plt.colorbar(orientation='horizontal',pad=0.10)
        cb.ax.tick_params(labelsize=13)
        xtick_marks = np.arange(len(xLabel))
        ytick_marks = np.arange(len(yLabel))
        plt.xticks(xtick_marks, xLabel,fontsize = 12)
        plt.yticks(ytick_marks, yLabel,fontsize = 12)

        for i, j in itertools.product(range(len(yLabel)), range(len(xLabel))):
            plt.text(j, i, "{:0.4f}".format(data[dataset][i][j]),
                        horizontalalignment="center",
                        color="black", fontsize= 13)
        plt.tight_layout()

        plt.title(dataset + " dataset",fontsize = 12)

        plt.savefig('./plot/exp3_' + dataset + '.png')
        plt.show()