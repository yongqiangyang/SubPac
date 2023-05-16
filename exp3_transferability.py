import subprocess
import os
import math
import csv
import random
import time
import plot.exp3
from utils import get_change_rate, test, run

if __name__ == "__main__":
    poisonMethods = ["POAC"]
    subMethods = ["MV", "WMV", "EM", "ZenCrowd"]
    attackedMethods = ["MV", "WMV", "ZenCrowd", "c_EM", "CATD", "GLAD", "KOS", "VI-BP"]
    datasets = ["temp", "rte", "sentiment"]
    mWorkers = 5
    answerPercents = [0.5]
    knowPercents = [1.0]
    seeds = 1
    exp = "exp3"
    
    run(poisonMethods, subMethods, attackedMethods, datasets, mWorkers, answerPercents, knowPercents, seeds, exp)
    
    plot.exp3.plot(subMethods, attackedMethods)