Revised for Milestone 5


Background
==========
Our task is to predict the veracity of a given rumor. The rumor is in the form of a tweet that reports an update associated with a newsworthy event. The labels describe whether the rumor is true or false, as well as a confidence percentage between 0 and 1. There are two variants to this task. In subtask A, the veracity of a rumour will have to be predicted solely from the tweet itself (closed variant).


In subtask B, predicting veracity is dependent on the provided additional context. This context consists of relevant snapshots, including a snapshot of an associated Wikipedia article, a Wikipedia dump, news articles from digital news outlets retrieved from NewsDiffs, as well as preceding tweets from the same event. These supplementary data are labelled in **/data** as *'[support]'*; no other external information is necessary.

More information of the task can be found here: https://competitions.codalab.org/competitions/16172#learn_the_details


- The training data can be found in ./semeval2017-task8-dataset*
- The dev data can be found in ./data*
- The test data can be found in ./semeval2017-task8-test-data*.



Instructions
============


Dependencies
------------

Must install via pip3 or conda:
- gensim
- sklearn
- numpy
- pandas
- nltk
- jupyter notebook

Given its large size, the glove word2vec file cannot be versioned. It must be placed manually in the top project directory, and should be named glove.6B.200d.txt.word2vec. It can be downloaded here: http://nlp.stanford.edu/data/glove.6B.zip


To run unit tests:
------------------

$ python3 test_runner.py


To train and evaluate:
----------------------

Instructions on how to run each of these notebooks are included in each notebook!

To train non-RNN classifiers, add custom features, and evaluate the model:
    `$ jupyter notebook main.ipynb`

To train various the RNN, add custom features, and evaluate the model:
    `$ jupyter notebook RNN.ipynb`

To demonstrate tweet text run on an RNN (akin to Homework 6), and evaluate:
	`$ jupyter notebook RNN_Text.ipynb`



