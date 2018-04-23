from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.models import KeyedVectors
from gensim.models import Word2Vec
import string
import json
import pickle
jstr = ""

# glove_input_file = 'glove.6B/glove.6B.50d.txt'
# word2vec_output_file = 'glove.6B.50d.txt.word2vec'
# glove2word2vec(glove_input_file, word2vec_output_file)
########## NOTES:
########## glove.6B.50d.txt.word2vec is much faster, so it might be easier to use 
########## that for testing purposes. This can be done by using EmbedExtractor(True).
########## Otherwise, the paper specifies 200 dimensions, so use glove.6B.200d.txt.word2vec.

class EmbedExtractor():
	def __init__(self,testing=False):
		# load the Stanford GloVe model, which has been converted to W2V
		size=200
		if testing:
			size=50
		filename = 'glove.6B.'+str(size)+'d.txt.word2vec'
		self.model = KeyedVectors.load_word2vec_format(filename, binary=False)
		self.wv = self.model.wv
		self.size = size

	def tweetVec(self, tweet):
		pad=[0.0]*self.size
		translator = str.maketrans('', '', string.punctuation)
		vec=[]
		tweetParse=[word.translate(translator).lower() for word in tweet.split(" ")]
		tweetParse=[word for word in tweetParse if word!=""]
		for word in tweetParse:
			try:
				vec.extend(self.wv[word])
			except KeyError:
					vec.extend(pad)
		wordsLeft=45-len(tweetParse)  #I assume that most tweets will have less than 45 words.
		for i in range(wordsLeft):
			vec.extend(pad)
		return vec

ee = EmbedExtractor()

for d in ["train","dev","test"]:
	with open('./output/full/'+d+'_data_full.json', 'r') as f:
	    jstr = f.read()

	j = json.loads(jstr)
	tweet2vec = {}

	for key in j:
		tweet=j[key]['text']
		tweet2vec[key]=ee.tweetVec(tweet)

	pickle.dump(tweet2vec,open(d+"_word_embedding_vectors.pickle","wb"))

