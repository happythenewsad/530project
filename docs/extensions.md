Extensions
==========

Extensions are not completed unless marked COMPLETED


- additional features       COMPLETED    
  'num_replies', 
  're_has_?', 
  're_has_NOT', 
  're_has_correct',
  're_has_credib', 
  're_has_data', 
  're_has_detail', 
  're_has_fabricat', 
  're_has_lie', 
  're_has_proof', 
  're_has_source', 
  're_has_witness', 
  'opinion', 
  'user_default_profile',
  'user_favourites_count', 
  'user_followers_count', 
  'user_friends_count', 
  'user_geo_enabled', 
  'user_listed_count', 
  'user_statuses_count', 
  'user_verified', 
  'user_created'

- RNN model

- Random Forest model

- Ensemble model(s)
  - combinations of NB, SVM, RF, etc

- dimensionality reduction (specifically on word embedding features)



Scores:
=======

public baseline to beat: 28.57%


| Remark | Test accuracy |
|--------|---------------|
| **PRE-PCA** |               |
| Naive Bayes, no 'containsURL', embeddings       |               0.50|
| Naive Bayes, no 'containsURL', no embeddings       |               0.25|
| Naive Bayes, no 'containsURL', no embeddings, user, reply       |               0.50|
| **Naive Bayes, no embeddings, no user, reply**      | **0.53** |
| SVM       |        0.32       |
| SVM, no 'containsURL'       |        0.32       |
| Decision Tree       |     0.28          |
| Decision Tree, no 'containsURL'       |     0.32          |
|  RNN, no 'containsURL', no embeddings   |    0.28           |
|  RF, default parameters  |    0.21           |
|  RF, n_samples = 20  |    0.25           |
|  RF, n_samples = 40  |    0.28           |
|  RF, n_samples = 80  |    0.32           |
|  RF, n_samples = 80, optimal feature set (row 4)  |    0.25           |
|  |               |
| POST-PCA |               |
| Naive Bayes, optimal feature set (row 4) | 0.39 |
| SVM, optimal feature set (row 4)  |    0.35          |
| **Decision Tree, optimal feature set (row 4)**  |    **0.53**    <sup>1</sup>      |
| Random Forest, n_samples = 80, optimal feature set (row 4)  |    0.28          |

1. The score for the decision tree fluctuates depending on the initialization of the tree








