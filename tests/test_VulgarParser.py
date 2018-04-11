import unittest

from VulgarExtractor import VulgarExtractor



class Tester(unittest.TestCase):
    def test_vulgar(self):
        vulgarWords = VulgarExtractor.vulgarWords("badwords.txt") 

        nonVulgarLine = "foo bar"
        self.assertEqual(VulgarExtractor.containsVulgar(nonVulgarLine, vulgarWords), False)

        vulgarLine = "foo ass"
        self.assertEqual(VulgarExtractor.containsVulgar(vulgarLine, vulgarWords), True)