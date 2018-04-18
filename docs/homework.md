Milestone 4 
===========


Task
----

RumorEval: Determining rumour veracity and support for rumours


Motivation
----------

Rumor evaluation has substantial overlap with the current fake news problem. It would be great if we could identify fake news automatically.

Example tweet:

Coup in #Russia? Good article by @Forbes. http:\/\/t.co\/aaNCpb0blW #RamzanKadyrov #Putin #putindead http:\/\/t.co\/OfivdTlTbO

Is this a rumor or not? This tweet references a Forbes article, a normally trustworthy news source. But notice how the tweet provides commentary that goes farther than the article. Namely, that Putin is in fact dead, not merely absent.


Task instantiated as research problem
-------------------------------------

SemEval ‘17: ‘RumourEval’ Task B - Closed
https://competitions.codalab.org/competitions/16173


Scoring
-------

How should we evaluate the sucess of our models? Confidence weighted accuracy seems intuitively better than binary accuracy, since less confident predictions could be transformed by a downstream system for better use-specific performance. 

Not all models used included confidence values, so we focused on recording and optimizing binary accuracy.

Investigating other scoring metrics may prove fruitful, but this is out of scope of the project.


Baselines
---------

Stupid baseline (majority class labeling): 0.48
Best public baseline*: 28.57% (IITP)

Extensions and exploration
--------------------------

We decided to try new models and new features:

Models
  - RNN, using the scalar features only

Techniques
  - dimensionality reduction
  - Feature generation
  - Parameter tweaking

Feature generation (expanded)
  - Text characteristic features used by IITP and NileTMRG
    e.g. Word count, POS tags, MPQA subjectivity lexicon, vulgar words
  - Tweet metadata
    Number of followers, account creation date, contains images
  - Context features
    Replies, Wikipedia articles


Results and Analysis
--------------------

Best performing model
  - Naive Bayes classifier (as expected, since IITP's best classifier was Naive Bayes)
  - Text features
  - Tweet replies: how ‘skeptical’ were the readers’ reactions?
    11 features that gauged the proportion of replies containing “source?”, “proof?”, “lie!”, etc.

Score: 0.53 (accuracy)

We created 20 additional custom features. We selected 11 of these for use in the final model by intuitive experimentation. We found that dropping word embedding features increased the Naive Bayes model accuracy. This is problem because the cardinality of the word embedding features is quite high, and thus drowns out the predictive power of the other features. A natural mitigation would be to apply dimensionality reduction to the word embedding features. We plan to try this in future experiments.


One of the first things we noticed about our benchmark paper was that their score against dev data was significantly lower than their score against test data. This implied that excessive tuning against dev data caused overfitting in their models. Thus, we only recorded test scores that were not lower than the corresponding dev scores. Further evidence of overfitting on IITP's end included testing on the dev-set and obtaining accuracy scores signifcantly lower than theirs, despite implementing all of their said features. 



Things we didn't try, but wanted to
-----------------------------------

Feature weighting

In-depth feature extraction from replies

Deep feature extraction from linked articles

Other models:
  - Ensemble
  - RandomForest


