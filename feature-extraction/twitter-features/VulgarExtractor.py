import re

class VulgarExtractor:

	@staticmethod
	def vulgarWords(filename="vulgar_words.txt"):
	    words = []
	    with open(filename, 'r') as f:
	        for line in f:
	            words.append(line.strip())
	    return words

	@staticmethod
	def containsVulgar(line, wordlist):
	    words = re.split(r'\W+', line)
	    for word in words:
	        if word in wordlist:
	            return True
	    return False   