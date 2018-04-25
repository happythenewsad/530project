import unittest

from VulgarExtractor import VulgarExtractor



class Tester(unittest.TestCase):
    def test_vulgar(self):
        # this path param is relative to test_runner.py
        vulgarWords = VulgarExtractor.vulgarWords("./code/feature-extraction/vulgar-extractor/badwords.txt") 

        nonVulgarLine = "foo bar"
        self.assertEqual(VulgarExtractor.containsVulgar(nonVulgarLine, vulgarWords), False)

        vulgarLine = "foo ass"
        self.assertEqual(VulgarExtractor.containsVulgar(vulgarLine, vulgarWords), True)