import pickle
import json
from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.models import KeyedVectors
from gensim.models import Word2Vec
import string
jstr = ""

class EmbedExtractor():
	def __init__(self,testing=False):
		# load the Stanford GloVe model, which has been converted to W2V
		size=200
		if testing:
			size=50
		filename = 'glove.6B.'+str(size)+'d.txt.word2vec'
		self.model = KeyedVectors.load_word2vec_format(filename, binary=False)
		self.wv = self.model.wv

	def tweetVec(self, tweet):
		pad=[0.0]*len(self.wv["the"])
		translator = str.maketrans('', '', string.punctuation)
		vec=[]
		tweetParse=[word.translate(translator).lower() for word in tweet.split(" ")]
		tweetParse=[word for word in tweetParse if word!=""]
		for word in tweetParse:
			try:
				vec.extend(self.wv[word])
			except KeyError:
					vec.extend(pad)
		wordsLeft=30-len(tweetParse)  #I assume that most tweets will have less than 30 words.
		for i in range(wordsLeft):
			vec.extend(pad)
		return vec


with open('./output/full/dev_data_full.json', 'r') as f:
    jstr = f.read()

j = json.loads(jstr)
ee = EmbedExtractor()

tweet2vec = {}

for key in j:
	tweet=j[key]['text']
	tweet2vec[key]=tweet

pickle.dump(tweet2vec,open("feature_embedding.pickle","wb"))

