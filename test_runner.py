import os
import unittest
import logging
import sys

sys.path.insert(1, "./feature-extraction/twitter-features")

all_tests = unittest.TestLoader().discover(os.path.join(os.path.dirname(__file__), "tests"), "test_*.py")
unittest.TextTestRunner(verbosity=2).run(all_tests)