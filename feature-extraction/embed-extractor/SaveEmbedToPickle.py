import pickle
import json
from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.models import KeyedVectors
from gensim.models import Word2Vec
import string
from EmbedExtractor import EmbedExtractor
jstr = ""


with open('./output/full/dev_data_full.json', 'r') as f:
    jstr = f.read()

j = json.loads(jstr)
ee = EmbedExtractor()

tweet2vec = {}

for key in j:
	tweet=j[key]['text']
	tweet2vec[key]=tweet

pickle.dump(tweet2vec,open("feature_embedding.pickle","wb"))