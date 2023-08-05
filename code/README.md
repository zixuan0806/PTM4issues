# Paper Name

This repo is the code for the paper "paper name".

## Quick Start

### Dependencies

```shell
pip install -r requirements.txt
pip install xlrd openpyxl
```

### Code structure

1. `my_data` dir contains the prepared data.
2. `output` dir contains the output log.
3. `get_issues.ipynb` contains the code for crawling the github issue data.
4. `data_preparation.ipynb` contains the code for issue data preprocessing.
5. `text_augment.py` contains the code for data augment we use in the data preprocessing phase.
6. `train.py` contains the main training code for our experiments.
7. `run_cross.sh` is the scripts for train on one repo and test on another.
8. `run.sh` is the scripts for train different models on specific repo.

### Run Instruction

Because it takes a long time to label the data, So we apply the label data at `code/my_data`

We have prepared a script for an easy run. To test the code, simply input the following command...

1. run experiments for train cross repo(replace the repo in the comment to run different trials).

```shell
cd code
nohup .run_cross.sh &
```

2. run experiments for different models

```shell
cd code
nohup .run.sh &
```

