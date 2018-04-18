FOR DEVELOPER USE ONLY

badwords.txt is generated from https://gist.githubusercontent.com/jamiew/1112488/raw/7ca9b1669e1c24b27c66174762cb04e14cf05aa7/google_twunter_lol

badwordsformatter.py formats the original input into a 1 word per line text file.


"feature-extraction/twitter-features/word_embedding_vectors.pickle" contains all the word embedding vectors in the form {"tweet_id":[vector]}. 


A paragraph for the discussions section - yd
A big challenge of the task was the inherent difficulty of identifying a false rumor. During the feature selection process, we found that some tweets were difficult to classify even using human judgment on the full data. For instance, one tweet (552978184413921281) posted by a former journalist is an image supposedly created by the graffiti artist Banksy regarding the Charlie Hebdo attack. This tweet is labeled false, as the illustration was misattributed - it was a work of another artist unrelated to the incident. However, the data provided does not provide any information regarding the artwork and it is impossible to verify the truth with only the data given. 
Another example (id 524947867975561216) is a tweet posted by cnn's official account. The text content of the tweet is as follows: 'Ottawa Police Service: There were "numerous gunmen" at the Canada War Memorial shooting. One person was shot. http://t.co/zNhxK6wBoy'.
While the tweet was made by a reputable source and the text was innocuous in content, this data point is labeled false, due to there having been only one gunman in the scene, not multiple; the rest of the tweet is true. While this information can be verified by examining the supplementary wikipedia context data, the page itself links to another news article that reports 'multiple gunmen' being at the scene. These examples suggest that detecting some false rumors require a semantic level decisions based on very minor details.
They also raise awareness on the need for a more specific goal in our classification task. Not all fake news are created equal: there are varying degrees of falsehood and the underlying intent also plays a role. Does having a mild inaccuracy in an initial news report warrant the same label as a malicious fabricated story designed to mislead readers? What about news content that intentionally omit facts for an agenda? Should it be considered to be true? 
These questions will prove beneficial in designing future systems for detecting falsehood in social media.
