import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot():
    sns.set()
    plt.switch_backend('agg')

    plt.rcParams['figure.figsize'] = (16.0, 3.0)
    plt.rcParams['savefig.dpi'] = 300 
    plt.rcParams['figure.dpi'] = 300 
    plt.figure()

    x1=[0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45]

    # POAC
    base_folder = './output/exp1/'
    POAC = []
    datasets = ['temp', 'rte', 'sentiment']
    for dataset in datasets:
        temp =[]
        with open(base_folder + "change_rate_" + dataset + "_POAC") as f:
            text = f.read().split(',')
            for i in range(0, 7):
                temp.append(float(text[i]))
            POAC.append(temp)
            f.close()

    # TIA
    TIA = []
    for dataset in datasets:
        temp =[]
        with open(base_folder + "change_rate_" + dataset + "_TIA") as f:
            text = f.read().split(',')
            for i in range(0, 7):
                temp.append(float(text[i]))
            TIA.append(temp)
            f.close()

    # Flip
    Flip = []
    for dataset in datasets:
        temp =[]
        with open(base_folder + "change_rate_" + dataset + "_Flip") as f:
            text = f.read().split(',')
            for i in range(0, 7):
                temp.append(float(text[i]))
            Flip.append(temp)
            f.close()

    # Rand
    Rand = []
    for dataset in datasets:
        temp =[]
        with open(base_folder + "change_rate_" + dataset + "_Rand") as f:
            text = f.read().split(',')
            for i in range(0, 7):
                temp.append(float(text[i]))
            Rand.append(temp)
            f.close()
    legendFontsize = 12
    # first picture
    plt.subplot(1, 3, 1)
    plt.plot(x1,POAC[0],'ro-',label = "SubPac")
    plt.plot(x1,TIA[0],'*-',label = "TIA")
    plt.plot(x1,Flip[0],'^-',label = "Flip")
    plt.plot(x1,Rand[0],'x-',label = "Rand")
    plt.tick_params(axis='both',which='major',labelsize=12,pad = 1.0,direction = 'in')
    plt.title('Temp dataset',fontsize = 17)
    plt.xlabel('Percentage of instances',fontsize = 17)
    plt.ylabel('Attack success rate',fontsize = 17,labelpad = 1.0)
    plt.legend(fontsize = legendFontsize)
    plt.tight_layout()
    # second picture
    plt.subplot(1, 3, 2)
    plt.plot(x1,POAC[1],'ro-',label = "SubPac")
    plt.plot(x1,TIA[1],'*-',label = "TIA")
    plt.plot(x1,Flip[1],'^-',label = "Flip")
    plt.plot(x1,Rand[1],'x-',label = "Rand")
    plt.tick_params(axis='both',which='major',labelsize=12,pad = 1.0,direction = 'in')
    plt.title('Rte dataset',fontsize = 17)
    plt.xlabel('Percentage of instances',fontsize = 17)
    plt.ylabel('Attack success rate',fontsize = 17,labelpad = 1.0)
    plt.legend(fontsize = legendFontsize)
    plt.tight_layout()
    # third picture
    plt.subplot(1, 3, 3)
    plt.plot(x1,POAC[2],'ro-',label = "SubPac")
    plt.plot(x1,TIA[2],'*-',label = "TIA")
    plt.plot(x1,Flip[2],'^-',label = "Flip")
    plt.plot(x1,Rand[2],'x-',label = "Rand")
    plt.tick_params(axis='both',which='major',labelsize=12,pad = 1.0,direction = 'in')
    plt.title('Sentiment dataset',fontsize = 17)
    plt.xlabel('Percentage of instances',fontsize = 17)
    plt.ylabel('Attack success rate',fontsize = 17,labelpad = 1.0)
    plt.legend(fontsize = legendFontsize)
    plt.tight_layout()

    plt.savefig('./plot/exp1.png')
