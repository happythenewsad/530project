Revised for Milestone 4


Background
==========
Our task is to predict the veracity of a given rumor. The rumor is in the form of a tweet that reports an update associated with a newsworthy event. The labels describe whether the rumor is true or false, as well as a confidence percentage between 0 and 1. There are two variants to this task. In subtask A, the veracity of a rumour will have to be predicted solely from the tweet itself (closed variant).


In subtask B, predicting veracity is dependent on the provided additional context. This context consists of relevant snapshots, including a snapshot of an associated Wikipedia article, a Wikipedia dump, news articles from digital news outlets retrieved from NewsDiffs, as well as preceding tweets from the same event. These supplementary data are labelled in **/data** as *'[support]'*; no other external information is necessary.

More information of the task can be found here: https://competitions.codalab.org/competitions/16172#learn_the_details


- The training data can be found in ./semeval2017-task8-dataset*
- The dev data can be found in ./data*
- The test data can be found in ./semeval2017-task8-test-data*.


The baseline (./notebooks/baseline.json) assumes with full confidence that all the rumors are false. This results in a 48% accuracy — roughly equal to randomly guessing. **scorer** returns the accuracies, as well as whether or not each entry was labeled correctly.

Instructions
============

To generate the stupid baseline (in place of simple-baseline.py): 
    `$ jupyter notebook notebooks/exploration.ipynb`

To train the Naive Bayes model, add custom features, and evaluate the model:
    `$ jupyter notebook main.ipynb`



