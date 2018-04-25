README for outputs/
===================


Evaluation script (this script is called by various notebooks):
---------------------------------------------------------------
    `$ python3 scorer/score.py LABELED_DATA.json PREDICTED_DATA.json` 


This example runs the test predictions against gold test labels:
---------------------------------------------------------------
    `$ python3 scorer/score.py goldtest.json output/classifier_output/goldtest_nb.json`


* the test predictions JSON file must already be generated
