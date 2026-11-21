import utils
import unittest
from os.path import join

from arekit.common.data.storages.base import BaseRowsStorage
from arekit.common.data.rows_fmt import create_base_column_fmt
from arekit.common.data.rows_parser import ParsedSampleRow

from arelight.arekit.api.data_type import DataType
from arelight.arekit.parsed_row_service import ParsedSampleRowExtraService
from arelight.arekit.samples_io import CustomSamplesIO
from arelight.readers.csv_pd import PandasCsvReader


class TestAREkitIterData(unittest.TestCase):

    samples_target_func = lambda data_type: join(
        utils.TEST_DATA_DIR, "-".join(["arekit-iter-data", data_type.name.lower()]))

    def test(self):
        samples_io = CustomSamplesIO(create_target_func=TestAREkitIterData.samples_target_func,
                                     reader=PandasCsvReader(sep=',', compression=None, custom_extension=".csv"))
        samples_filepath = samples_io.create_target(data_type=DataType.Test)
        samples = samples_io.Reader.read(samples_filepath)
        assert(isinstance(samples, BaseRowsStorage))

        column_fmts = [create_base_column_fmt(fmt_type="parser")]

        for ind, sample_row in samples:
            parsed_row = ParsedSampleRow(sample_row, columns_fmts=column_fmts, no_value_func=lambda: None)
            source = ParsedSampleRowExtraService.calc("SourceValue", parsed_row=parsed_row)
            target = ParsedSampleRowExtraService.calc("TargetValue", parsed_row=parsed_row)
            print(source, target)

