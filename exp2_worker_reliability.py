import subprocess
import os
import math
import csv
import random
import time
import plot.exp2
from utils import get_change_rate, test, run

if __name__ == "__main__":
    poisonMethods = ["POAC", "TIA", "Flip", "Rand"]
    subMethods = ["EM"]
    attackedMethods = ["c_EM"]
    datasets = ["temp", "rte", "sentiment"]
    mWorkers = 2
    answerPercents = [0.0]
    knowPercents = [1.0]
    seeds = 1
    exp = "exp2"

    # run(poisonMethods, subMethods, attackedMethods, datasets, mWorkers, answerPercents, knowPercents, seeds, exp)
    
    plot.exp2.plot()