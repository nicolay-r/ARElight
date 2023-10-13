import utils
import unittest
from os.path import join

from arekit.common.pipeline.base import BasePipeline
from arekit.common.data.storages.base import BaseRowsStorage
from arekit.common.experiment.data_type import DataType
from arekit.contrib.networks.input.rows_parser import ParsedSampleRow
from arekit.contrib.utils.data.readers.csv_pd import PandasCsvReader
from arekit.contrib.utils.io_utils.samples import SamplesIO

from arelight.arekit.parsed_row_service import ParsedSampleRowExtraService
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

        for ind, sample_row in samples:
            parsed_row = ParsedSampleRow(sample_row)
            source = ParsedSampleRowExtraService.calc("SourceValue", parsed_row=parsed_row)
            target = ParsedSampleRowExtraService.calc("TargetValue", parsed_row=parsed_row)
            print(source, target)

    def test_pipeline_item(self):
        samples_io = SamplesIO(target_dir=utils.TEST_DATA_DIR,
                               reader=PandasCsvReader(sep=',', compression=None, custom_extension=".csv"),
                               prefix="arekit-iter-data",
                               writer=None)

        pipeline = BasePipeline(pipeline=[
            D3jsGraphsBackendPipelineItem()
        ])

        ppl_result = PipelineResult({
            "d3js_graph_output_dir": utils.TEST_OUT_DIR,
            "d3js_host": 8000,
        })
        ppl_result.update("samples_io", samples_io)
        ppl_result.update("predict_filepath", value=join(utils.TEST_OUT_DIR, "predict.tsv.gz"))
        pipeline.run(input_data=ppl_result)

