badwords.txt is generated from https://gist.githubusercontent.com/jamiew/1112488/raw/7ca9b1669e1c24b27c66174762cb04e14cf05aa7/google_twunter_lol

badwordsformatter.py formats the original input into a 1 word per line text file.


"feature-extraction/twitter-features/word_embedding_vectors_dev.pickle" contains all the word embedding vectors in the form {"tweet_id":[vector]}. 


EmbedExtractor.py requires you to download gensim, which you can do with "pip3 install gensim"
