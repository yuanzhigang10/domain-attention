# Domain Attention

Domain Attention Model proposed in paper *"Domain Attention Model for Multi-Domain Sentiment Classification"* ([link](https://www.sciencedirect.com/science/article/pii/S0950705118302144)), which is accepted by [*Knowledge Based Systems*](https://www.journals.elsevier.com/knowledge-based-systems/).

## Data

[Multi-Domain Sentiment Dataset (version 2.0)](https://www.cs.jhu.edu/~mdredze/datasets/sentiment/)

The dataset we used is [unprocessed.tar.gz](https://www.cs.jhu.edu/~mdredze/datasets/sentiment/unprocessed.tar.gz) which contains the original data.

## Train & Test

For example, you can use the following command to test the model in the MDSD dataset:

> python model.py

The parameter settings are list in `SharedData` which is defined in `model.py`.

The run-log files of 3 repeated runs (using the default parameters in the scripts) can be found in `/run-logs` folder, which give the following accuracy results:

| Domain | Books | DVD | Electronics | Kitchen |
|:-----:|:-----:|:-----:|:-----:|:-----:|
| Run1 | 0.8762 | 0.8663 | 0.8663 | 0.8911 |
| Run2 | 0.8861 | 0.8614 | 0.8713 | 0.8960 |
| Run3 | 0.8960 | 0.8614 | 0.8713 | 0.8960 |

The results reported in the paper are averaged through ten runs (10-fold cross validation), which maybe slightly different from the above results.

Notice: you should first download the MDSD dataset and [glove](https://nlp.stanford.edu/projects/glove/) word vectors to run this code. The glove version we used is [glove.6B.zip](http://nlp.stanford.edu/data/glove.6B.zip).

## Code Description

 - `data_helper.py` data loader for the MDSD dataset
 - `model.py` the domain attention model, train and test
 - `utils.py` misc util functions
 
 
For any issues, you can contact me via [yuanzg14@mails.tsinghua.edu.cn](mailto:yuanzg14@mails.tsinghua.edu.cn)
