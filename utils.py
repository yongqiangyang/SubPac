import sys
import numpy as np
import configparser
import uuid
import json
from tqdm import tqdm
import csv
import os
import random
import math
import time
import subprocess
def loadData(dataset, proportion):
    # print("Loading data...\n")
    worker_instance_num = {'sentiment': 235, 'rte': 49, 'temp': 61, 'ducks': 108, 'ER': 142, 'SPC': 137}
    DATA_PATH = r'./datasets/' + dataset + '/'
    ground_truth = np.load(DATA_PATH + 'label_train.npy', allow_pickle=True)
    raw_answers = np.load(DATA_PATH + 'data_answer.npy', allow_pickle=True)
    raw_ins_num = raw_answers.shape[0] 
    worker_num = raw_answers.shape[1] 
    random_missing = random.sample(range(0, raw_ins_num), int(raw_ins_num * (1 - proportion)))
    ins_num = raw_ins_num - int(raw_ins_num * (1 - proportion))
    answers = np.ones((ins_num, worker_num)) * -1
    num = 0
    for i in range(raw_ins_num):
        if i not in random_missing:
            for j in range(worker_num):
                answers[num][j] = raw_answers[i][j]
            num += 1 
    
    answers = answers.astype(np.int32)

    normal_workers_aggregation = []
    for i in range(ins_num):
        votes = np.zeros(2)
        for r in range(worker_num):
            if answers[i, r] != -1:
                votes[answers[i, r]] += 1
        normal_workers_aggregation.append(np.argmax(votes))
    # print("Loading completed.\n")
    return normal_workers_aggregation, ground_truth, answers, raw_ins_num, ins_num, worker_num, random_missing, worker_instance_num[dataset]

def gete2wlandw2el_normal(answers):
    e2wl = {}
    w2el = {}
    label_set=['0','1']
    
    for i in range(answers.shape[0]):
        for j in range(answers.shape[1]):
            if answers[i][j] != -1:
                example, worker, label = str(i),str(j),str(answers[i][j])
                if example not in e2wl:
                    e2wl[example] = []
                e2wl[example].append([worker,label])

                if worker not in w2el:
                    w2el[worker] = []
                w2el[worker].append([example,label])

    return e2wl,w2el,label_set
    
def gete2wlandw2el(answers, answers_p):
    e2wl = {}
    w2el = {}
    label_set=['0','1']
    
    for i in range(answers.shape[0]):
        for j in range(answers.shape[1]):
            if answers[i][j] != -1:
                example, worker, label = str(i),str(j),str(answers[i][j])
                if example not in e2wl:
                    e2wl[example] = []
                e2wl[example].append([worker,label])

                if worker not in w2el:
                    w2el[worker] = []
                w2el[worker].append([example,label])
    
    for i in range(answers_p.shape[0]):
        for j in range(answers_p.shape[1]):
            if answers_p[i][j] != -1:
                example, worker, label = str(i),str(j),str(answers_p[i][j])
                worker_p = str(int(worker) + answers.shape[1])
                if example not in e2wl:
                    e2wl[example] = []
                e2wl[example].append([worker_p,label])

                if worker_p not in w2el:
                    w2el[worker_p] = []
                w2el[worker_p].append([example,label])

    return e2wl,w2el,label_set

def writeTotalAnswers(answers, dataset, poisonMethod, subMethod, mWorkers, answerPercent, knowPercent, seed):
    folder = r'./outputPoisonAnswer/' + dataset
    if not os.path.isdir(folder):
        os.mkdir(folder)
    folder = r'./outputPoisonAnswer/' + dataset + r'/' + poisonMethod
    if not os.path.isdir(folder):
        os.mkdir(folder)
    folder = r'./outputPoisonAnswer/' + dataset + r'/' + poisonMethod + r'/' + subMethod
    if not os.path.isdir(folder):
        os.mkdir(folder)
    poisonDataFile = r'./outputPoisonAnswer/' + dataset + r'/' + poisonMethod + r'/' + subMethod + r'/mW_' + str(mWorkers) + '-answerPercent_' + str(answerPercent) + '-knowPercent_' + str(knowPercent) + '-s_' + str(seed) + '.csv'
    with open(poisonDataFile, "w") as w:
        writer = csv.writer(w)
        writer.writerows([["question", "worker", "answer"]])
        [rows, cols] = answers.shape
        for row in range(rows):
            for col in range(cols):
                if (answers[row, col] != -1):
                    writer.writerows(
                        [[row, col, answers[row, col]]])
    return poisonDataFile

def get_change_rate(truth_file, output_normal, output_normal_and_malicious, method):
    # When normal workers make the right questions, the rate at which malicious workers change it
    arr0 = [-1] * len(output_normal)
    f = open(truth_file, 'r')
    reader = csv.reader(f)
    next(reader)
    for line in reader:
        example, label = line
        arr0[int(example)] = int(label)

    arr1 = [-1] * len(output_normal)
    arr2 = [-1] * len(output_normal)

    for key in output_normal:
        ind = int(key)
        if method == "CATD":
            arr1[ind] = int(output_normal[key])
        else:
            if '0' not in output_normal[key]:
               output_normal[key]['0'] = 1.0 - output_normal[key]['1']
            if '1' not in output_normal[key]:
               output_normal[key]['1'] = 1.0 - output_normal[key]['0']                
            if output_normal[key]['0'] > output_normal[key]['1']:
                arr1[ind] = 0
            else:
                arr1[ind] = 1

    for key in output_normal_and_malicious:
        ind = int(key)
        if method == "CATD":
            arr2[ind] = int(output_normal_and_malicious[key])
        else:
            if '0' not in output_normal_and_malicious[key]:
               output_normal_and_malicious[key]['0'] = 1.0 - output_normal_and_malicious[key]['1']
            if '1' not in output_normal_and_malicious[key]:
               output_normal_and_malicious[key]['1'] = 1.0 - output_normal_and_malicious[key]['0']
            
            if output_normal_and_malicious[key]['0'] > output_normal_and_malicious[
                    key]['1']:
                arr2[ind] = 0
            else:
                arr2[ind] = 1

    numerator = sum(a == b and b != c for a, b, c in zip(arr0, arr1, arr2))
    denominator = sum(a == b for a, b in zip(arr0,arr1))
    
    return float(numerator / (denominator * 1.0))

def test(dataset, attackedMethod, poisonDataFile):
    normalAnswerFile = "./datasets/" + dataset + "/answer.csv"
    output_normal = subprocess.getoutput("python " + r'./attackedMethods/' + attackedMethod + r'/method.py ' + normalAnswerFile + ' ' + '"categorical"').split('\n')[-1]
    output_normal_and_malicious = subprocess.getoutput("python " + r'./attackedMethods/' + attackedMethod + r'/method.py ' + poisonDataFile + ' ' + '"categorical"').split('\n')[-1]
    truth_file = r"./datasets/" + dataset + r"/truth.csv"
    change_rate = get_change_rate(truth_file, eval(output_normal), eval(output_normal_and_malicious),str(attackedMethod))
    return change_rate

def run(poisonMethods, subMethods, attackedMethods, datasets, mWorkers, answerPercents, knowPercents, seeds, exp):
    for poisonMethod in poisonMethods:
        for dataset in datasets:
            print()
            change_rates = []
            for subMethod in subMethods:
                for attackedMethod in attackedMethods:
                    for answerPercent in answerPercents:
                        for knowPercent in knowPercents:
                            change_rate = 0
                            for seed in range(1, seeds + 1):
                                poisonDataFile = r'./outputPoisonAnswer/' + dataset + r'/' + poisonMethod + r'/' + subMethod + r'/mW_' + str(mWorkers) + '-answerPercent_' + str(answerPercent) + '-knowPercent_' + str(knowPercent) + '-s_' + str(seed) + '.csv'
                                change_rate += test(dataset, attackedMethod, poisonDataFile)
                            print(" dataset: %-11s poisonMethod: %-10s subMethod: %-10s attackedMethod: %-10s answerPercent: %.2f knowPercent: %.2f seed: %d change_rate: %f" % (dataset, poisonMethod, subMethod, attackedMethod, answerPercent, knowPercent, seed, change_rate / seeds))
                            change_rates.append(str(change_rate / seeds))
            folder = r'./output/' + exp
            if not os.path.isdir(folder):
                os.mkdir(folder)
            f = open(folder + '/' + 'change_rate_' + dataset + '_' + poisonMethod, 'w')
            for change_rate in change_rates:
                f.write(change_rate + ',')
            f.close()
