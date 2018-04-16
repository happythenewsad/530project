import sys
#sys.path.insert(1, "./feature-extraction/embed-extractor/")
#from EmbedExtractor import EmbedExtractor
from FileReader import FileReader
sys.path.insert(1, "./feature-extraction/vulgar-extractor/")
from VulgarExtractor import VulgarExtractor
sys.path.insert(1, "./feature-extraction/opinion-extractor/")
from OpinionExtractor import OpinionExtractor
sys.path.insert(1, "./feature-extraction/twitter-parser/")
from TwitterParser import TwitterParser

import classifiers


def normalize(column_name):
    std = df[column_name].std()
    norm_col = df[column_name].apply(lambda x: x - std)
    df[column_name] = norm_col

def create_opinion_column(df):
    #add a binary column where opinion == 1 if the tweet text contains a strongly subjective word
    global strongly_subj_list
    OpinionExtractor.add_opinion_column(df, strongly_subj_list)

def create_vulgar_column(df):
    global wordlist
    dftext = df[['text']]
    result = dftext.applymap(lambda x: VulgarExtractor.containsVulgar(x,wordlist))
    df['isVulgar'] = result


if __name__ == "__main__":
#    exec(open('FileReader.py').read())

    full_df_list = FileReader.get_dataframe()

    #load strongly subjective list
    strongly_subj_list = OpinionExtractor.initialize_subjectivity()

    #load vulgar words list
    wordlist = VulgarExtractor.vulgarWords("./feature-extraction/vulgar-extractor/badwords.txt") 


    #this loop will generate all features
    for df in full_df_list:
        create_opinion_column(df)
        create_vulgar_column(df)

    
    train_df = full_df_list[0]
    dev_df = full_df_list[1]
    test_df = full_df_list[2]
  
    #print the first line of each dataframe to check output
    print(train_df.iloc[0])
    print(dev_df.iloc[0])
    print(test_df.iloc[0])


#TODO - modularize methods below and add them into the for loop above
"""
ee = EmbedExtractor()

#word embeddings must be generated before POS
word_embeddings = [ee.tweetVec(tagged_line) for tagged_line in df['text']]

textlist = [txt.replace('\n','') for txt in df['text'].tolist()]
tagged_sents = TwitterParser.tag(textlist)
df['POS'] = tagged_sents
#df.head()

processed_sents = []
for tagged_sent in df['POS']:
    processed_words = []
    for word, tag in tagged_sent:
        if tag == 'U':
            processed_words.append('someurl')
        elif tag == '@':
            processed_words.append('@someuser')
        else:
            processed_words.append(word)
    sent = ' '.join(processed_words)
    processed_sents.append(sent)
df['text'] = processed_sents

# import TwitterParser features
word_counts = [TwitterParser.word_count(tagged_line) for tagged_line in df['POS']]
pos_count_list = [TwitterParser.pos_counts(tagged_line) for tagged_line in df['POS']]
contains_adjs = [TwitterParser.contains_adjectives(tagged_line) for tagged_line in df['POS']]
contains_urls = [TwitterParser.contains_url(tagged_line) for tagged_line in df['POS']]
contains_emojis = [TwitterParser.contains_emoji(tagged_line) for tagged_line in df['POS']]
contains_abbrevs = [TwitterParser.contains_abbreviation(tagged_line) for tagged_line in df['POS']]

# get word count and normalize
df['wordCount'] = word_counts
normalize('wordCount')

# get an indexed list of pos tag counts
df['posCounts'] = pos_count_list

# get binary features
df['containsAdjective'] = contains_adjs
df['containsURL'] = contains_urls
df['containsEmoji'] = contains_emojis
df['containsAbbreviation'] = contains_abbrevs
df['wordEmbedding'] = word_embeddings

for i, tag in enumerate(TwitterParser.tagset):
    tag_counts = []
    for pos_counts in df['posCounts']:
        tag_counts.append(pos_counts[i])
    column_name = 'num_' + tag
    df[column_name] = tag_counts
    normalize(column_name)


df.loc[df.classification == 'true', 'classification'] = 1
df.loc[df.classification == 'false', 'classification'] = 0
df.loc[df.classification == 'unverified', 'classification'] = 2


attributes = []
# getting the labels
# You have to comment this out if you want only tweet ID to be in the features. 
# Note that by doing this, you will screw up the simple/tr,simple/dev test located after this


attributes = ['isVulgar', 'containsAdjective', 'containsURL', 'containsEmoji', 'containsAbbreviation', 'wordCount']
for tag in TwitterParser.tagset:
    attributes.append('num_' + tag)
    
dev_labels = df['classification']
dev_labels = [l for l in dev_labels]
dev_labels = np.array(dev_labels)

# getting the values as a list of lists
dev_values = df[attributes].values.tolist()
word_embedding_values = df['wordEmbedding'].values.tolist()

#Below puts the tweet ID as a feature. Comment this out if you aren't using tweetID
###for i,index in enumerate(df.index):
###    dev_values[i].append(int(index))



for i,d in enumerate(word_embedding_values):
    dev_values[i].extend(d)
    
dev_values = np.array(dev_values, dtype=object)




# note predict_proba() gets probabilities for all 3 labels
#... and decision_tree_classifier uses decision_function() instead of predict_proba()... weird sklearn quirk
# NOTE: This is where you change the classifier type. Can pick from [naive_bayes, svm_classifier, decision_tree_classifier]
predictions, probabilities = classifiers.naive_bayes(dev_values, dev_labels, dev_values)

ps = []
for i, p in enumerate(predictions):
    if p == 0:
        ps.append('false')
    if p == 1:
        ps.append('true')
    if p == 2:
        ps.append('unverified')
    

# creates pairings of the prediction and the probability of the prediction
pred_probs_pairs = [[ps[i], probabilities[i][predictions[i]]] for i in range(len(predictions))]  

#now we make a dictionary of tweetID to the pred_probs_pairs
pred_dict = {index:pred_probs_pairs[i] for i,index in enumerate(df.index)}
#pred_dict


#python3 scorer/score.py semeval2017-task8-dataset/traindev/rumoureval-subtaskB-dev.json output/classifier_output/test.json



with open('output/classifier_output/test.json', 'w') as outfile:
    json.dump(pred_dict, outfile)


# Some comments about performance: WE SHOULD BE SEEING 100% percent veracity accuracy (when using our regular features) naive_bayes = .76 veracity accuracy 
# svm = 1.00 veracity accuracy 
# decision tree = 1.00 veracity accuracy 

# (when using nothing but tweetID)
# naive_bayes = .44 veracity accuracy
# svm = .48 veracity accuracy 
# decision tree = 1.00 veracity accuracy



# Training on simple/tr, testing on simple/dev

# Simply doing all of the transformations we did in the first few cells

tr_df = pd.read_pickle('output/simple/train_data_simple.pickle')

sys.path.insert(1, "./feature-extraction/twitter-features")

wordlist = VulgarExtractor.vulgarWords("badwords.txt") 
dftext = tr_df[['text']]
result = dftext.applymap(lambda x: VulgarExtractor.containsVulgar(x,wordlist))
tr_df['isVulgar'] = result

word_embeddings = [ee.tweetVec(tagged_line) for tagged_line in tr_df['text']]
textlist = [txt.replace('\n','') for txt in tr_df['text'].tolist()]
tagged_sents = TwitterParser.tag(textlist)
tr_df['POS'] = tagged_sents

processed_sents = []
for tagged_sent in tr_df['POS']:
    processed_words = []
    for word, tag in tagged_sent:
        if tag == 'U':
            processed_words.append('someurl')
        elif tag == '@':
            processed_words.append('@someuser')
        else:
            processed_words.append(word)
    sent = ' '.join(processed_words)
    processed_sents.append(sent)
tr_df['text'] = processed_sents

word_counts = [TwitterParser.word_count(tagged_line) for tagged_line in tr_df['POS']]
pos_count_list = [TwitterParser.pos_counts(tagged_line) for tagged_line in tr_df['POS']]
contains_adjs = [TwitterParser.contains_adjectives(tagged_line) for tagged_line in tr_df['POS']]
contains_urls = [TwitterParser.contains_url(tagged_line) for tagged_line in tr_df['POS']]
contains_emojis = [TwitterParser.contains_emoji(tagged_line) for tagged_line in tr_df['POS']]
contains_abbrevs = [TwitterParser.contains_abbreviation(tagged_line) for tagged_line in tr_df['POS']]

tr_df['wordCount'] = word_counts
tr_df['posCounts'] = pos_count_list
tr_df['containsAdjective'] = contains_adjs
tr_df['containsURL'] = contains_urls
tr_df['containsEmoji'] = contains_emojis
tr_df['containsAbbreviation'] = contains_abbrevs
tr_df['wordEmbedding'] = word_embeddings


for i, tag in enumerate(TwitterParser.tagset):
    tag_counts = []
    for pos_counts in tr_df['posCounts']:
        tag_counts.append(pos_counts[i])
    column_name = 'num_' + tag
    tr_df[column_name] = tag_counts
    normalize(column_name)


# Changes "true"/"false"/"unverified" to numeric values, just like the in the early cells

tr_df.loc[tr_df.classification == 'true', 'classification'] = 1
tr_df.loc[tr_df.classification == 'false', 'classification'] = 0
tr_df.loc[tr_df.classification == 'unverified', 'classification'] = 2
# getting the labels

attributes = ['isVulgar', 'containsAdjective', 'containsURL', 'containsEmoji', 'containsAbbreviation', 'wordCount']
for tag in TwitterParser.tagset:
    attributes.append('num_' + tag)

tr_labels = tr_df['classification']
tr_labels = [l for l in tr_labels]
tr_labels = np.array(tr_labels)


# getting the values as a list of lists
tr_values = tr_df[attributes].values.tolist()
word_embedding_values = tr_df['wordEmbedding'].values.tolist()


#Below puts the tweet ID as a feature. Comment this out if you aren't using tweetID
###for i,index in enumerate(df.index):
###    dev_values[i].append(int(index))


for i,d in enumerate(word_embedding_values):
    tr_values[i].extend(d)
    
tr_values = np.array(tr_values)


predictions, probabilities = classifiers.svm_classifier(tr_values, tr_labels, dev_values)
print(probabilities)
ps = []
for i, p in enumerate(predictions):
    if p == 0:
        ps.append('false')
    if p == 1:
        ps.append('true')
    if p == 2:
        ps.append('unverified')
    

# creates pairings of the prediction and the probability of the prediction
pred_probs_pairs = [[ps[i], probabilities[i][predictions[i]]] for i in range(len(predictions))]  


pred_dict = {index:pred_probs_pairs[i] for i,index in enumerate(df.index)}


#pred_dict

with open('output/classifier_output/tr_test.json', 'w') as outfile:
    json.dump(pred_dict, outfile)
    
#test with:
### python3 scorer/score.py semeval2017-task8-dataset/traindev/rumoureval-subtaskB-dev.json output/classifier_output/tr_test.json



# Some comments about performance:
# naive_bayes = .32 veracity accuracy 
# svm = .24 veracity accuracy 
# decision tree = .36 veracity accuracy
"""
