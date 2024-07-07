import unittest

from tqdm import tqdm

from arelight.arekit.sample_service import AREkitSamplesService


class TestIterSamplesSqliteStorage(unittest.TestCase):

    def test(self):

        # Compose joined table.

        # Read joined table.
        data_it = AREkitSamplesService.iter_samples_and_predict_sqlite3(
            sqlite_filepath="../output/example.txt-test",
            samples_table_name="contents",
            predict_table_name="open_nre_bert",
            filter_record_func=None)

        for data in tqdm(data_it):
            print(data)