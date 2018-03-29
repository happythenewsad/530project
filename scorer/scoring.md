Evaluation
==============
From the challenge:

"The evaluation of the predicted veracity, which will be one of true or false for each instance, will be performed with microaveraged accuracy, hence measuring the ratio of instances for which a correct prediction is made."

Additional metrics, precision, recall, and F1 score were added. Higher scores for all of these metrics correlate with better performance.

The rsme is the only metric that takes into account levels confidence, which the machine uses whenever making a claim that a certain text contains factual data.
The lower this score, the better.

Scoring can be done by using the following bash command:

python scorer/score.py {truth values file.json} {submission file.json}