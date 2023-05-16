import subprocess
import os
import math
import csv
import random
import time
import plot.exp1
from utils import get_change_rate, test, run

if __name__ == "__main__":
    poisonMethods = ["POAC", "TIA", "Flip", "Rand"]
    subMethods = ["EM"]
    attackedMethods = ["c_EM"]
    datasets = ["temp", "rte", "sentiment"]
    mWorkers = 5
    answerPercents = [0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45]
    knowPercents = [1.0]
    seeds = 1
    exp = "exp1"

    run(poisonMethods, subMethods, attackedMethods, datasets, mWorkers, answerPercents, knowPercents, seeds, exp)
    
    
    plot.exp1.plot()