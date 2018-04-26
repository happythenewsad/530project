# RumourEval: Determining rumour veracity and support for rumours
## Authors: Yoonduk Kim, Peter Kong, Sam Korn, Jason Tang, Noah Weiner

---

### Abstract
The goal of the RumourEval task is to determine the veracity of rumours in Twitter posts about newsworthy events. The project is based on the [SemEval '18 Task 8](https://competitions.codalab.org/competitions/16173), by the same name. We implemented the [best public baseline](http://www.derczynski.com/sheffield/papers/rumoureval/171_Paper.pdf), published by a team from the Indian Institute of Technology Patna, with an accuracy of 28.57%. We then designed a number of extensions to the baseline, including a recurrent neural net, PCA dimensionality reduction, and skepticism keyword features from reply tweets. With these additional features, we were able to increase the accuracy of our best model to 53.57%.

---

### Literature Review

The only team that had participated in subtask B-Closed was IITP (Singh et al., 2017). The team uses the Naive Bayes classifier (which they reported to yield the best performance) after generating features from the text contents of the tweet. The text is preprocessed to normalize all urls and user references. The battery of text features include word embedding, vulgar words, presence of metadata, word count, POS tag, negation words, wh- words, opinion words as obtained from MPQA subjectivity lexicon, and the number of adjectives.

IITP qualified for the "open" subtask by including a binary variable that indicated whether the tweet had media context to support its claims. It should be noted that the team had the lowest score among the 5 participants for the "closed" subtask at 0.286 and their "open" model performed marginally better, with the score of 0.393. This leaves open much room for improvement, especially since the tweets' source material and the reply chain had not been explored.

To gain insight into task B, we reviewed work by teams [IKM (Chen et al., 2017)](http://www.derczynski.com/sheffield/papers/rumoureval/53_Paper.pdf) and [NileTMRG (Enayet and El-Beltagy, 2017)](http://www.aclweb.org/anthology/S17-2082), who tied for first place on subtask B-open with the score of 0.536. IKM implemented a convolutional neural network, using the word embedding matrices generated from the tweets as features. The CNN is composed of a convolution layer, a pooling layer, and a final fully connected layer with a softmax activation function. The results display the potential of neural network models, which may be further investigated in our main study.

Meanwhile, NileTMRG took an approach similar to IITP, where they used a linear SVM model on the following text features: question existence, denial terms, support words, hashtags, urls, whether the tweet is a reply, tweet sentiment. In addition, they also utilized the tweet’s metadata, such as user verification, number of friends/followers/past tweets, retweet ratio, existence of profile photo, days since user creation, and whether the source tweet was verified. The addition of metadata features contributed in improving the performance of the model by a significant amount.

In our research, we first replicate the IITP model as our baseline for Rumoreval subtask B-open. We then improve the baseline by adding metadata features used by NileTMRG and the context features, as well as using PCA dimensionality reduction to shorten the feature vectors for the word embeddings, and an RNN trained on both text and context features.

---

### Experimental Design

#### Data

Our data for this project was taken directly from the SemEval task data. The data consists of three data sets: training, development, and testing.

Each of the data sets contains a relatively small number of data points, though each data point contains a large amount of information. The number of data points in each data set is summarized below:

| Data Set    | # of Data Points|
|:-----------:|:---------------:|
| Training    | 272             |
| Development | 25              |
| Testing     | 28              |

Each data point consists of a string literal of the text of the tweet, as well as a host of other metadata. Each data point also examines all of the reply tweets, which contain a similar set of metadata. The metadata is stored as a JSON file, and describes the number of followers of the user, whether or not the user is verified, URLS in the tweet, and much more. An example metadata file is included in the Appendix.

The label for each individual data point is stored in a separate JSON file, which maps tweet IDs to the string "true", "false", or "unverified".


#### Evaluation Metric

The evaluation metric specificed by the SemEval task is pure accuracy:

$$ acc(N) = \frac{(t\_{true} + t\_{false} + t\_{unverified})}{|N|} $$

Here, $$t\_{label}$$ represents the number of data points in the evaluated data set correctly labeled as *label* and $$|N|$$ represents the total number of data points in the evaluated data set.

We considered other potential evaluation metrics, including confidence-weighted accuracy and precision-based F1 score, but we chose to designate these as out of scope of the project, since there were no comparable scores in the literature.


#### Simple Baseline

For a simple baseline, we used majority class labeling, which finds the label assigned to the largest number of data points in the training data set, and assigns every data point that label. Applied to the testing data set, this baseline returned an accuracy score of 48%.

---

### Experimental Results

#### Published Baseline

As previously mentioned, the published baseline we implemented was created by a team at the Indian Institute of Technology Patna (or IITP). The baseline used a Naive Bayes classifier and several text-based feature, as well as a metadata feature.

##### Features:
1. Word Embeddings

   The word embeddings were pretrained 200-dimensional [GloVe vectors](https://nlp.stanford.edu/pubs/glove.pdf), which were converted to the word2vec format. Each word in the sentence was embedded and concatenated. The sentence was padded out to a maximum sentence length.

2. POS Features

   The [CMU ARK-Tweet NLP Twitter Tagger](http://www.cs.cmu.edu/~ark/TweetNLP/) was used to tokenize and POS tag the twitter posts. From this several features were extracted:
   - Word Count
   - Contains adjective
   - Contains URL
   - Contains abbreviation
   - Contains emoji

3. Vulgarity Feature

   The vulgarity feature determined the "vulgarity" of the tweet text. A list of "vulgar" words was extracted from [Google's List of Bad Words](http://fffff.at/googles-official-list-of-bad-words/), and a binary feature for presence or absence of a word from the list was implemented.

4. Subjectivity Feature

   The subjectivity feature determined the "subjectivity" of the tweet text. A list of highly subjective words was extracted from the [MPQA Subjectivity Lexicon](http://people.cs.pitt.edu/~wiebe/pubs/papers/emnlp05polarity.pdf), and a binary feature for presence or absence of a word from the list was implemented.

5. Metadata Feature

   The metadata feature indicated the presence or absence of metadata, such as URLs, images, and reply tweets.

Implementing our interpretation of the published baseline achieved the same accuracy score of 28%.


#### Extensions

We implemented a number of extensions to improve on the baseline score, including an RNN, PCA dimensionality reduction, reply features, and user features. With all of our models (except the RNN), we used the Naive Bayes classifier, which consistently outperformed other classifiers.

##### RNN
A recurrent neural network framework, supplied by the PyTorch library, was used in experimentation to address several concerns regarding the clashing of data types and size of the dataset.

The first action taken was building a recurrent neural network that took in a combination of the most prevalent binary and scalar features. Two issues came out of this. The first was the difficulty arranging the included data tro fit the serialized structure that most RNN inputs take. The second was the size of the dataset, as there were only slightly below 300 examples given by RumorEval.

The second RNN model that was used solely used the text of the tweet and was built in such a way that was similar to the homework 6 implementation. Scores for both of these models were unimpressive due to the intuitive concerns mentioned previously. Furthermore, the second model gave more evidence that tweet text does not provide substantial evidence towards one of the two definitive classifications. For example, many topics require a posteriori knowledge in order to make any rational judgment and it can be said with confidence that fake tweets, including those given in the dataset, are disguised very well.

##### PCA Dimensionality Reduction
In the original paper, the tweet was represented word by word using word embeddings derived from the Stanford 200d GloVe set. Tweets were then padded to ensure a consistent length. We set a generous upper limit of 45 words and concatenated the features, resulting in a 9000x1 feature vector. This led to a gross imbalance of features in our feature vector, which was 30-40 features long without the word embedding, and significantly decreased our performance. To mitigate this, we used principal component analysis to reduce the feature vectors to 24, which was the upper estimate of the average number of words. This still reduced the performance, albeit by only 3%.

##### Reply Features
This feature set, which proved incredibly valuable, examined the the degree of skepticism found in tweets that replied to the original post. This was measured by the percentage of tweets that contained a number of different "skeptical" words such as "witness", "lie", and "proof".

##### User Features
This feature set extracted several relevant data points from the rich context data provided in the "Open" variation of the Task B data set. Some of the extracted information included the number of followers, whether the account was verified, whether the account was geo-enabled, and several others.

##### Performance on Test Data Set:
| Model            | Accuracy |
|:----------------:| --------:|
| Baseline         | 28%      |
| RNN              | 28%      |
| Baseline + PCA   | 50%      |
| Baseline + Reply | **53%**  |
| Baseline + User  | 35%      |


#### Error Analysis

**Confusion matrix**

| Labels     | True | Unverified | False |
|------------|------|------------|-------| 
| True       |   5  |      1     |   2   | 
| Unverified |   0  |      4     |   4   | 
| False      |   4  |      2     |   6   | 

*The rows and columns correspond to the prediction and true labels, respectively.*

The confusion matrix shows that the 13 misclassified examples (from the total of 28) are evenly divided into false positives (7) and false negatives (6). Taking a further look into the test examples, we observe the following pattern:

All false negative data points (tweets that were labeled “false” when they were true or unverified) come from the topics that were covered in the training data. Out of the 6 tweets, 3 are from the Ottawa shootings, 2 from the Germanwings crash, and 1 from Charlie Hebdo. This indicates that the classifier is more pessimistic for topics that are familiar to the model. An interesting point to note is that all 6 tweets are related to death. The wording of the tweets and the replies are therefore more emotional. The classifier may have interpreted the somber contents to be indicators of falsehood based on the trained features.

On the other hand, out of the 7 false positive tweets only 1 is from the topic included in the training data (Charlie Hebdo). The other 6 are from an entirely different topic on Hillary Clinton’s health concerns. In the one Charlie Hebdo example, the focus of the tweet is on the closure of shops in the neighborhood, not of the deaths. The comparatively more neutral vocabulary used for these tweets may have lead the model to classify them as true, as the contents were less emotionally charged.

This shed light on the importance of recognizing the context when performing evaluations on truthfulness. Some topics tend to be more emotional while others are lighter in tone; this inherent difference may lead to misclassification, especially when the model is trained on the existence of emotionality and opinion. Therefore, a useful extension to the model would be first identifying the topic of the tweet and measuring the deviance from the average sentiment of the given topic. However, this would require a much greater training dataset composed of tweets of various topics with varying degrees of emotion.

---

### Conclusions

One of the conclusions we came to over the course of the project was that our baseline, recreated from the IITP whitepaper, was a very poor model. There was significant overfitting, likely due to the high dimensionality of the word embedding vectors, which were 9000 dimensions, and overshadowed the other features.

Another conclusion we reached was that the sizes of the data sets provided by the SemEval task were too small for meaningful testing. With a test dataset size of 28 data points, a single data point could determine whether a feature increased or decreased the test score. As such, we noticed a small amount of overfitting, not to the training data, but to the test data, by the means of our feature selection which was tailored to improving test accuracy.

Finally, we see a number of potential future steps for this task:
- Feature weighting
- In-depth feature extraction from replies
- Deep feature extraction from linked articles
- Other classifiers:
  - Ensemble classifiers
  - Convolutional (instead of recurrent) neural net

---

### Acknowledgements

We would like to acknowledge the initial findings of the research team at the Indian Institute of Technology Patna, for there baseline whitepaper. Additionally, we would like to thank our advisor, Anne Cocos, for her mentorship and advice.

---

### Appendix: Tweet Metadata Example

```json
{
	"contributors":null,
	"truncated":false,
	"text":"AC Milan midfielder Michael Essien has been diagnosed with Ebola. Get well soon Michael. [Daily Times] http:\/\/t.co\/r6y8d9HMAw",
	"in_reply_to_status_id":null,
	"id":521360486387175424,
	"favorite_count":328,
	"source":"<a href=\"http:\/\/twitter.com\/download\/iphone\" rel=\"nofollow\">Twitter for iPhone<\/a>",
	"retweeted":false,
	"coordinates":null,
	"entities":{
		"symbols":[],
		"user_mentions":[],
		"hashtags":[],
		"urls":[],
		"media":[{
			"source_status_id_str":"521359454672269313",
			"expanded_url":"http:\/\/twitter.com\/TransferRelated\/status\/521359454672269313\/photo\/1",
			"display_url":"pic.twitter.com\/r6y8d9HMAw",
			"url":"http:\/\/t.co\/r6y8d9HMAw",
			"media_url_https":"https:\/\/pbs.twimg.com\/media\/Bzw9qTNCQAE0a0c.jpg",
			"source_status_id":521359454672269313,"id_str":"521359453786882049",
			"sizes":{
				"large":{"h":330,"resize":"fit","w":600},
				"small":{"h":187,"resize":"fit","w":340},
				"medium":{"h":330,"resize":"fit","w":600},
				"thumb":{"h":150,"resize":"crop","w":150}
			},
			"indices":[103,125],
			"type":"photo",
			"id":521359453786882049,
			"media_url":"http:\/\/pbs.twimg.com\/media\/Bzw9qTNCQAE0a0c.jpg"
		}]
	},
	"in_reply_to_screen_name":null,
	"id_str":"521360486387175424",
	"retweet_count":781,
	"in_reply_to_user_id":null,
	"favorited":false,
	"user":{
		"follow_request_sent":false,
		"profile_use_background_image":true,
		"default_profile_image":false,
		"id":1594903015,
		"profile_background_image_url_https":"https:\/\/abs.twimg.com\/images\/themes\/theme1\/bg.png",
		"verified":false,
		"profile_text_color":"333333",
		"profile_image_url_https":"https:\/\/pbs.twimg.com\/profile_images\/580854919856537600\/LC9U-g5i_normal.jpg",
		"profile_sidebar_fill_color":"DDEEF6",
		"entities":{
			"url":{
				"urls":[{
					"url":"http:\/\/t.co\/DdcwSaKAKQ",
					"indices":[0,22],
					"expanded_url":"http:\/\/bit.ly\/1w0PD0h",
					"display_url":"bit.ly\/1w0PD0h"
				}]
			},
			"description":{"urls":[]}
		},
		"followers_count":115868,
		"profile_sidebar_border_color":"C0DEED",
		"id_str":"1594903015",
		"profile_background_color":"C0DEED",
		"listed_count":84,
		"is_translation_enabled":false,
		"utc_offset":3600,
		"statuses_count":728,
		"description":"The Futbol Life. It's More Than A Game. Bringing You The Latest Football News, Views, Transfers And More. Business Enquiries: Futbol.Life@outlook.com",
		"friends_count":153,
		"location":"Worldwide",
		"profile_link_color":"0084B4",
		"profile_image_url":"http:\/\/pbs.twimg.com\/profile_images\/580854919856537600\/LC9U-g5i_normal.jpg",
		"following":false,
		"geo_enabled":false,
		"profile_banner_url":"https:\/\/pbs.twimg.com\/profile_banners\/1594903015\/1411166936",
		"profile_background_image_url":"http:\/\/abs.twimg.com\/images\/themes\/theme1\/bg.png",
		"screen_name":"FutbolLife",
		"lang":"en",
		"profile_background_tile":false,
		"favourites_count":6,
		"name":"FutbolLife",
		"notifications":false,
		"url":"http:\/\/t.co\/DdcwSaKAKQ",
		"created_at":"Mon Jul 15 03:32:28 +0000 2013",
		"contributors_enabled":false,
		"time_zone":"London",
		"protected":false,
		"default_profile":true,
		"is_translator":false
	},
	"geo":null,
	"in_reply_to_user_id_str":null,
	"possibly_sensitive":false,
	"lang":"en",
	"created_at":"Sun Oct 12 18:03:21 +0000 2014",
	"in_reply_to_status_id_str":null,
	"place":null,
	"extended_entities":{
		"media":[{
			"source_status_id_str":"521359454672269313",
			"expanded_url":"http:\/\/twitter.com\/TransferRelated\/status\/521359454672269313\/photo\/1",
			"display_url":"pic.twitter.com\/r6y8d9HMAw",
			"url":"http:\/\/t.co\/r6y8d9HMAw",
			"media_url_https":"https:\/\/pbs.twimg.com\/media\/Bzw9qTNCQAE0a0c.jpg",
			"source_status_id":521359454672269313,
			"id_str":"521359453786882049",
			"sizes":{
				"large":{"h":330,"resize":"fit","w":600},
				"small":{"h":187,"resize":"fit","w":340},
				"medium":{"h":330,"resize":"fit","w":600},
				"thumb":{"h":150,"resize":"crop","w":150}
			},
			"indices":[103,125],
			"type":"photo",
			"id":521359453786882049,
			"media_url":"http:\/\/pbs.twimg.com\/media\/Bzw9qTNCQAE0a0c.jpg"
		}]
	}
}
```
