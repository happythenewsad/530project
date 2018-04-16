Our task is to predict the veracity of a given rumor. The rumor is in the form of a tweet that reports an update associated with a newsworthy event. The labels describe whether the rumor is true or false, as well as a confidence percentage between 0 and 1. There are two variants to this task. In subtask A, the veracity of a rumour will have to be predicted solely from the tweet itself (closed variant).

In subtask B, predicting veracity is dependent on the provided additional context. This context consists of relevant snapshots, including a snapshot of an associated Wikipedia article, a Wikipedia dump, news articles from digital news outlets retrieved from NewsDiffs, as well as preceding tweets from the same event. These supplementary data are labelled in **/data** as *'[support]'*; no other external information is necessary.

More information of the task can be found here: https://competitions.codalab.org/competitions/16172#learn_the_details

The training data can be found in **/semeval2017-task8-dataset**.
The dev data can be found in **/data**.
The test data can be found in **/semeval2017-task8-test-data**. 

The baseline (**/notebooks/baseline.json**) assumes with full confidence that all the rumors are false. This results in a 48% accuracy — roughly equal to randomly guessing. **scorer** returns the accuracies, as well as whether or not each entry was labeled correctly.

Update: By removing word embeddings (in which one word embedding's dimensions were added as features, resulting in a ~9000-D vector for each row), and containsURL, the Naive Bayes model gets .48 accuracy.


To generate the baseline (in place of simple-baseline.py): 
$ jupyter notebook notebooks/exploration.ipynb


Evaluation script (in place of scoring.md:
$ python3 scorer/scorerB.py LABELED_DATA.json PREDICTED_DATA.json 


A paragraph for the discussions section - yd
A big challenge of the task was the inherent difficulty of identifying a false rumor. For instance, one training example - id 524947867975561216 - is a tweet posted by cnn's official account. The text content of the tweet is as follows: 'Ottawa Police Service: There were "numerous gunmen" at the Canada War Memorial shooting. One person was shot. http://t.co/zNhxK6wBoy'.
While the tweet was made by a reputable source and the text was innocuous in content, this data point is labeled false, due to there having been only one confirmed gunman in the scene; the rest of the tweet is true. This information is only verified by examining the supplementary context information, a wikipedia page that states that there was only one gunman. 
The task is made more difficult as the wikipedia page itself links to another news article that reports 'multiple gunmen' being at the scene. This example suggest that detecting false rumors may require a semantic level information.
It also raises awareness about the need for a more specific goal in our classification task. Not all fake news are created equal: there are varying degrees of falsehood and the underlying intent also plays a role. Does having a mild inaccuracy in an initial news report warrant the same label as a malicious fabricated story designed to mislead readers? What about news content that intentionally omit facts for an agenda? Should it be considered to be true? 
These questions will prove beneficial in designing future systems for detecting falsehood in social media.
