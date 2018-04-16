import unittest

from FileReader import FileReader



class FileReaderTester(unittest.TestCase):
    def test_does_it_run(self):
        classInstance = FileReader()
        df = classInstance.exec()