# Black-Box Data Poisoning Attacks on Crowdsourcing (IJCAI23)
The GitHub repository for the paper "Black-Box Data Poisoning Attacks on Crowdsourcing" accepted by IJCAI 2023.

## 1 Introduction

This repository is built for the proposed method (SubPac). You can use anaconda's virtual environment to quickly reproduce the experiment results in this paper.

## 2 Usage

### 2.1 Requirement

+ linux
+ anaconda
+ matlab (Exp 3 is needed)

### 2.2 Setup Virtual Environment

First, You can use the spec-list file to quickly build a conda virtual environment and install the required packages.

 ```bash
 conda create  --name poisoning_attacks_on_crowdsourcing --file spec-list.txt
 conda activate poisoning_attacks_on_crowdsourcing
 ```

### 2.3 Recurrence experiment

Senond, You can execute the python file to reproduce the experiment.

```bash
python exp1_proportion_of_malicious_workers.py
python exp2_worker_reliability.py
python exp3_transferability.py
```

After running the python file, you can find the text-based experimental results in the `output` folder. Not only that, you can also find the graphical experimental results in the `plot` folder.

