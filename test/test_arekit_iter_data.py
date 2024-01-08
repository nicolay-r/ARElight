from arekit.common.data.rows_fmt import create_base_column_fmt
from arekit.common.data.rows_parser import ParsedSampleRow
from arekit.common.pipeline.base import BasePipelineLauncher

import utils
import unittest
from os.path import join

from arekit.common.data.storages.base import BaseRowsStorage
from arekit.common.experiment.data_type import DataType
from arekit.contrib.utils.data.readers.csv_pd import PandasCsvReader
from arekit.contrib.utils.io_utils.samples import SamplesIO

from arelight.arekit.parsed_row_service import ParsedSampleRowExtraService
from arelight.pipelines.demo.labels.formatter import CustomLabelsFormatter
from arelight.pipelines.demo.labels.scalers import CustomLabelScaler
from arelight.pipelines.demo.result import PipelineResult
from arelight.pipelines.items.backend_d3js_graphs import D3jsGraphsBackendPipelineItem


class TestAREkitIterData(unittest.TestCase):

    def test(self):
        samples_io = SamplesIO(target_dir=utils.TEST_DATA_DIR,
                               reader=PandasCsvReader(sep=',', compression=None, custom_extension=".csv"),
                               prefix="arekit-iter-data",
                               writer=None)
        samples_filepath = samples_io.create_target(data_type=DataType.Test)
        samples = samples_io.Reader.read(samples_filepath)
        assert(isinstance(samples, BaseRowsStorage))

        column_fmts = [create_base_column_fmt(fmt_type="parser")]

        for ind, sample_row in samples:
            parsed_row = ParsedSampleRow(sample_row, columns_fmts=column_fmts, no_value_func=lambda: None)
            source = ParsedSampleRowExtraService.calc("SourceValue", parsed_row=parsed_row)
            target = ParsedSampleRowExtraService.calc("TargetValue", parsed_row=parsed_row)
            print(source, target)

    def test_pipeline_item(self):
        samples_io = SamplesIO(target_dir=utils.TEST_DATA_DIR,
                               reader=PandasCsvReader(sep=',', compression=None, custom_extension=".csv"),
                               prefix="arekit-iter-data",
                               writer=None)

        pipeline = [
            D3jsGraphsBackendPipelineItem()
        ]

        ppl_result = PipelineResult({
            "labels_scaler": CustomLabelScaler(),
            "labels_formatter": CustomLabelsFormatter(),
            "d3js_graph_output_dir": utils.TEST_OUT_DIR,
            "d3js_host": None,
            "predict_reader": PandasCsvReader(compression='infer')
        })
        ppl_result.update("samples_io", samples_io)
        ppl_result.update("predict_filepath", value=join(utils.TEST_OUT_DIR, "predict.tsv.gz"))

        BasePipelineLauncher.run(pipeline=pipeline, pipeline_ctx=ppl_result, has_input=False)

