import unittest
from os.path import join

from tqdm import tqdm

from arelight.arekit.sample_service import AREkitSamplesService
from utils import TEST_DATA_DIR


class TestIterSamplesSqliteStorage(unittest.TestCase):

    def test(self):

        data_it = AREkitSamplesService.iter_samples_and_predict_sqlite3(
            sqlite_filepath=join(TEST_DATA_DIR, "samples_and_predict-test"),
            samples_table_name="contents",
            predict_table_name="open_nre_bert",
            filter_record_func=None)

        for data in tqdm(data_it):
            print(data)
