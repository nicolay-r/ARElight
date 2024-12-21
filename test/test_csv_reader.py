import unittest
from tqdm import tqdm
from os.path import join

import utils
from arelight.run.utils import iter_content
from arelight.utils import iter_csv_lines


class CsvReadingTest(unittest.TestCase):

    def test(self):
        file_path = join(utils.TEST_DATA_DIR, 'arekit-iter-data-test.csv')
        csv_file = open(file_path, mode="r", encoding="utf-8-sig")
        for line in iter_csv_lines(csv_file=csv_file, delimiter=',', column_name="text_a"):
            print(line)

    def test_iter_content(self):
        for _ in tqdm(iter_content("data/example.zip", csv_column="text_a", csv_delimiter=",")):
            pass


if __name__ == '__main__':
    unittest.main()
