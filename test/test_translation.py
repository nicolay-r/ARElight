import unittest

from arelight.third_party.googletrans import translate_value


class TestTranslation(unittest.TestCase):

    def test(self):
        x = translate_value("привет", dest="en", src="ru")
        print(x)
