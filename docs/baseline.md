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

Given its large size, the glove word2vec file cannot be version. It must be placed manually in the top project directory, and should be named glove.6B.200d.txt.word2vec


To run unit tests:
------------------

$ python3 test_runner.py


To train and validate model:
----------------------------

Open main.ipynb in jupyter and execute each cell in sequence.



Results (against dev set)
=======

Naive Bayes   = .32 veracity accuracy 
SVM           = .24 veracity accuracy 
Decision Tree = .36 veracity accuracy