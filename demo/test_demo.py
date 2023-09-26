import unittest
import process_data


class TestDemo(unittest.TestCase):

    def test(self):
        content = process_data.do_infer(None)
        print(content)

