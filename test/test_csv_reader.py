import unittest
from os.path import join

import utils
from arelight.run.utils import iter_csv_lines


class CsvReadingTest(unittest.TestCase):

    def test(self):
        file_path = join(utils.TEST_DATA_DIR, 'arekit-iter-data-test-0.csv')
        for line in iter_csv_lines(csv_filepath=file_path, delimiter=',', column_name="text_a"):
            print(line)


if __name__ == '__main__':
    unittest.main()
