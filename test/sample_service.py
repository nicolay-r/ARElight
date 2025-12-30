from arekit.common.data.rows_fmt import create_base_column_fmt
from arekit.common.data.rows_parser import ParsedSampleRow
from arekit.common.data.storages.base import BaseRowsStorage
from arekit.common.experiment.data_type import DataType
from arekit.common.labels.base import NoLabel

from arelight.arekit.custom_sqlite_reader import CustomSQliteReader
from arelight.arekit.joined_sqlite import JoinedSQliteBasedRowsStorage
from arelight.arekit.parsed_row_service import ParsedSampleRowExtraService
from arelight.arekit.samples_io import CustomSamplesIO
from arelight.pipelines.demo.labels.base import PositiveLabel, NegativeLabel
from arelight.pipelines.demo.labels.formatter import CustomLabelsFormatter


class AREkitSamplesService(object):

    @staticmethod
    def _extract_label_from_row(parsed_row):
        if parsed_row["col_0"] == 1:
            return NoLabel()
        elif parsed_row["col_1"] == 1:
            return PositiveLabel()
        elif parsed_row["col_2"] == 1:
            return NegativeLabel()

    @staticmethod
    def iter_samples_and_predict_sqlite3(sqlite_filepath, samples_table_name, predict_table_name,
                                         filter_record_func=None):
        assert(callable(filter_record_func) or filter_record_func is None)

        samples_io = CustomSamplesIO(
            create_target_func=lambda _: sqlite_filepath,
            reader=CustomSQliteReader(table_name_a=samples_table_name, table_name_b=predict_table_name,
                                      storage_type=JoinedSQliteBasedRowsStorage))

        samples_filepath = samples_io.create_target(data_type=DataType.Test)
        samples = samples_io.Reader.read(samples_filepath)
        assert (isinstance(samples, BaseRowsStorage))

        column_fmts = [
            # default parameters.
            create_base_column_fmt(fmt_type="parser"),
            # sentiment score.
            {"col_0": lambda value: int(value), "col_1": lambda value: int(value), "col_2": lambda value: int(value)}
        ]

        labels_fmt = CustomLabelsFormatter()

        for ind, sample_row in samples:

            parsed_row = ParsedSampleRow(sample_row, columns_fmts=column_fmts, no_value_func=lambda: None)

            # reading label.

            record = {
                "filename": sample_row["doc_id"].split(':')[0],
                "text": sample_row["text_a"].split(),
                "s_val": ParsedSampleRowExtraService.calc("SourceValue", parsed_row=parsed_row),
                "t_val": ParsedSampleRowExtraService.calc("TargetValue", parsed_row=parsed_row),
                "s_type": ParsedSampleRowExtraService.calc("SourceType", parsed_row=parsed_row),
                "t_type": ParsedSampleRowExtraService.calc("TargetType", parsed_row=parsed_row),
                "label": labels_fmt.label_to_str(AREkitSamplesService._extract_label_from_row(parsed_row))
            }

            if filter_record_func is None:
                yield record
            elif filter_record_func(record):
                yield record
