import unittest

import TwitterParser

def readVulgarWordsFile(filename="vulgar_words.txt"):
    words = []
    with open(filename, 'r') as f:
        for line in f:
            words.append(line)
    return words

def contains_vulgar(line, wordlist):
    for word, tag in tagged_line:
        if wordList.contains(word)
            return True
    return False   

class Tester(unittest.TestCase):
    def test_vulgar(self):
        tagged_line = ""
        self.assertEqual(TwitterParser.containsVular(tagged_line, True)