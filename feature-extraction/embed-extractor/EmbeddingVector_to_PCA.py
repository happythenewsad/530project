import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import decomposition
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
import pandas as pd
import string
import pickle 

def chunkIt(seq, num):
	avg = len(seq) / float(num)
	out = []
	last = 0.0
	zeros = [0]*int(avg)

	while last < len(seq):
		subVector = seq[int(last):int(last + avg)]
		if zeros==subVector and last>4700:
			break
		out.append(subVector)
		last += avg
	return out



for d in ["train","dev","test"]:
	pickleFile = d+"_word_embedding_vectors.pickle"
	word_embeddings = pd.read_pickle(pickleFile)


	word_embeddings_chunked = {}
	for word in word_embeddings:
		word_embeddings_chunked[word]=chunkIt(word_embeddings[word],30)
	word_embeddings_pca = {}
	for word in word_embeddings_chunked:
		pca = decomposition.PCA(n_components=24)
		x = np.array(word_embeddings_chunked[word])
		print(x)
		try:
			x_std = StandardScaler().fit_transform(x)
			pca.fit_transform(x_std)
			word_embeddings_pca[word]=pca.singular_values_
			# print(pca.singular_values_)
		except ValueError:
			# print(word_embeddings_chunked[word])
			print("UH OH")

	pickle.dump(word_embeddings_pca, open(d+"_pca_word_embedding_vector.pickle", "wb"))
