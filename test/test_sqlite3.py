import unittest
from os.path import join
from arelight.third_party.sqlite3 import SQLite3Service
from utils import TEST_DATA_DIR


class TestSQLite3(unittest.TestCase):

    def test_connect(self):
        service = SQLite3Service()
        service.connect(join(TEST_DATA_DIR, "opennre-data-test-predict.sqlite"))

        print("--------------")

        print(service.table_rows_count(table_name="contents"))

        print("--------------")

        print(service.get_column_names(table_name="contents"))

        print("--------------")

        for row in service.iter_rows(table_name="contents", return_dict=True, column_value="id", value="0"):
            print(row)

        print("--------------")

        for row in service.iter_rows(table_name="contents", return_dict=True):
            print(row)

        service.disconnect()
