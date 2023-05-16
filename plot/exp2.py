import numpy as np
import csv
import random
import matplotlib.pyplot as plt
import seaborn as sns

def plot():
    poisoning_methods = ["POAC", "TIA", "Flip", "Rand"]
    worker_num = {'sentiment': 85, 'rte': 164, 'temp': 76, 'ER': 176}
    datasets = ["temp", "rte", "sentiment"]
    # datasets = ["temp"]
    M_ps = 2
    seeds = 5
    instance_num = {'sentiment': 1000, 'rte': 800, 'temp': 462, 'ER': 8315}

    for poisoning_method in poisoning_methods:
        for dataset in datasets:
            truth_file = './datasets/' + dataset + '/truth.csv'
            truth = [-1] * instance_num[dataset]
            f = open(truth_file, 'r')
            reader = csv.reader(f)
            next(reader)
            for line in reader:
                example, label = line
                truth[int(example)] = int(label)
            worker_ability = [0.0] * int(worker_num[dataset] + M_ps)
            temp_worker_ability = [0.0] * int(worker_num[dataset] + M_ps)
            for seed in range(1, seeds + 1):
                datafile = './outputPoisonAnswer/'+ dataset + '/' + poisoning_method + '/EM/mW_2-answerPercent_0.0-knowPercent_1.0-s_' + str(seed) + '.csv'
                f = open(datafile, 'r')

                reader = csv.reader(f)
                next(reader)

                w2qa = {}
                answers = np.ones((instance_num[dataset],int(worker_num[dataset] + M_ps))) * -1
                for line in reader:
                    question,worker,answer = line
                    if worker not in w2qa:
                        w2qa[worker] = []
                    w2qa[worker].append([question,answer])

                for i in range(len(w2qa)):
                    for j in range(100):
                        random.seed(j)
                        sampling = random.sample(w2qa[str(i)],10)
                        molecular = 0.0
                        for question,answer in sampling:
                            if truth[int(question)] == int(answer):
                                molecular += 1.0
                        temp_worker_ability[i] += molecular / 10.0
            for i in range(len(worker_ability)):
                worker_ability[i] = temp_worker_ability[i] / (seeds * 100)
            datafile = r'./output/exp2/' + poisoning_method + '_' + dataset + r'_worker_ability.csv'
            with open(datafile, "w", newline='') as w:
                writer = csv.writer(w)
                temp = [["worker", "ability"]]
                writer.writerows(temp)
                for i in range(len(worker_ability)):
                    writer.writerows([[i,worker_ability[i]]])


    sns.set()
    plt.rcParams['figure.figsize'] = (12.0, 3.0)

    for dataset in datasets:
        plt.figure()
        plt.title(dataset + ' dataset', fontsize=10)
        datafile = r'./output/exp2/POAC_' + dataset + '_worker_ability.csv'
        f = open(datafile, 'r')
        x_1 = []
        y_1 = []
        x_p_1 = []
        y_p_1 = []
        reader = csv.reader(f)
        next(reader)
        num = 0
        for line in reader:
            num += 1
            worker, ability = line
            if num <= worker_num[dataset]:
                x_1.append(int(worker))
                y_1.append(float(ability))
            else:
                x_p_1.append(float(worker))
                y_p_1.append(float(ability))

        ## TIA
        datafile = r'./output/exp2/TIA_' + dataset + '_worker_ability.csv'
        f = open(datafile, 'r')
        x_2 = []
        y_2 = []
        x_p_2 = []
        y_p_2 = []
        reader = csv.reader(f)
        next(reader)
        num = 0
        for line in reader:
            num += 1
            worker, ability = line
            if num <= worker_num[dataset]:
                x_2.append(int(worker))
                y_2.append(float(ability))
            else:
                x_p_2.append(float(worker))
                y_p_2.append(float(ability))

        ## Flip
        datafile = r'./output/exp2/Flip_' + dataset + '_worker_ability.csv'
        f = open(datafile, 'r')
        x_3 = []
        y_3 = []
        x_p_3 = []
        y_p_3 = []
        reader = csv.reader(f)
        next(reader)
        num = 0
        for line in reader:
            num += 1
            worker, ability = line
            if num <= worker_num[dataset]:
                x_3.append(int(worker))
                y_3.append(float(ability))
            else:
                x_p_3.append(float(worker))
                y_p_3.append(float(ability))

        ## Rand
        datafile = r'./output/exp2/Rand_' + dataset + '_worker_ability.csv'
        f = open(datafile, 'r')
        x_4 = []
        y_4 = []
        x_p_4 = []
        y_p_4 = []
        reader = csv.reader(f)
        next(reader)
        num = 0
        for line in reader:
            num += 1
            worker, ability = line
            if num <= worker_num[dataset]:
                x_4.append(int(worker))
                y_4.append(float(ability))
            else:
                x_p_4.append(float(worker))
                y_p_4.append(float(ability))
        legendFontsize = 10
        plt.subplot(1, 4, 1)
        # plt.subplots_adjust(left=0.09, right=1, wspace=0.25, hspace=0.25, bottom=0.01, top=0.91)
        plt.ylim(0.0, 1.05)
        plt.tick_params(axis='both', which='major', labelsize=12, pad=1.0, direction='in')
        plt.title('SubPac', fontsize=17)
        plt.xlabel('Worker ID', fontsize=17)
        plt.ylabel('Worker reliability', fontsize=17, labelpad=1.0)
        plt.scatter(x_1, y_1, marker='o', alpha=0.6, label="Normal workers")
        plt.scatter(x_p_1, y_p_1, c='red', alpha=0.6, label="Malicious workers")
        plt.legend(fontsize=legendFontsize)
        plt.tight_layout()

        plt.subplot(1, 4, 2)
        plt.ylim(0.0, 1.05)
        plt.tick_params(axis='both', which='major', labelsize=12, pad=1.0, direction='in')
        plt.title('TIA', fontsize=17)
        plt.xlabel('Worker ID', fontsize=17)
        plt.ylabel('Worker reliability', fontsize=17, labelpad=1.0)
        plt.scatter(x_2, y_2, marker='o', alpha=0.6, label="Normal workers")
        plt.scatter(x_p_2, y_p_2, c='red', alpha=0.6, label="Malicious workers")
        plt.legend(fontsize=legendFontsize)
        plt.tight_layout()
        plt.subplot(1, 4, 3)
        plt.ylim(0.0, 1.05)
        plt.tick_params(axis='both', which='major', labelsize=12, pad=1.0, direction='in')
        plt.title('Flip', fontsize=17)
        plt.xlabel('Worker ID', fontsize=17)
        plt.ylabel('Worker reliability', fontsize=17, labelpad=1.0)
        plt.scatter(x_3, y_3, marker='o', alpha=0.6, label="Normal workers")
        plt.scatter(x_p_3, y_p_3, c='red', alpha=0.6, label="Malicious workers")
        plt.legend(fontsize=legendFontsize)
        plt.tight_layout()
        plt.subplot(1, 4, 4)
        plt.ylim(0.0, 1.05)
        plt.tick_params(axis='both', which='major', labelsize=12, pad=1.0, direction='in')
        plt.title('Rand', fontsize=17)
        plt.xlabel('Worker ID', fontsize=17)
        plt.ylabel('Worker reliability', fontsize=17, labelpad=1.0)
        plt.scatter(x_4, y_4, marker='o', alpha=0.6, label="Normal workers")
        plt.scatter(x_p_4, y_p_4, c='red', alpha=0.6, label="Malicious workers")
        plt.legend(fontsize=legendFontsize)
        plt.tight_layout()

        plt.savefig('./plot/exp2_' + dataset + '.png')
        plt.show()