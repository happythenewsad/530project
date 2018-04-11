badwords.txt is generated from https://gist.githubusercontent.com/jamiew/1112488/raw/7ca9b1669e1c24b27c66174762cb04e14cf05aa7/google_twunter_lol

badwordsformatter.py formats the original input into a 1 word per line text file.


"feature-extraction/twitter-features/word_embedding_vectors.pickle" contains all the word embedding vectors in the form {"tweet_id":[vector]}. 


Make sure that glove.6B.200d.txt.word2vec is present in the project directory, or the file_reader.ipynb will not work.