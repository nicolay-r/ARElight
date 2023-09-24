from os.path import dirname

from arekit.common.experiment.data_type import DataType
from arekit.common.pipeline.context import PipelineContext
from arekit.common.pipeline.items.base import BasePipelineItem
from arekit.contrib.networks.input.rows_parser import ParsedSampleRow
from arekit.contrib.utils.io_utils.samples import SamplesIO

from arelight.arekit.parsed_row_service import ParsedSampleRowExtraService
from arelight.backend.d3js.relations_graph_builder import make_graph_from_relations_array
from arelight.backend.d3js.relations_graph_operations import graphs_operations
from arelight.backend.d3js.ui_web_force import save_force_graph
from arelight.backend.d3js.ui_web_radial import save_radial_graph


class D3jsGraphsBackendPipelineItem(BasePipelineItem):

    def __init__(self, samples):
        assert(isinstance(samples, SamplesIO))
        self.__samples = samples

    @staticmethod
    def __iter_relations(samples):
        for _, sample_row in samples:
            parsed_row = ParsedSampleRow(sample_row)
            source = ParsedSampleRowExtraService.calc("SourceValue", parsed_row=parsed_row)
            target = ParsedSampleRowExtraService.calc("TargetValue", parsed_row=parsed_row)
            yield [source, target, "pos"]

    def iter_column_value(self, samples, column_value):
        for _, sample_row in samples:
            parsed_row = ParsedSampleRow({column_value: sample_row[column_value]})
            yield parsed_row[column_value]

    def apply_core(self, input_data, pipeline_ctx):
        assert(isinstance(pipeline_ctx, PipelineContext))

        target = pipeline_ctx.provide("backend_template")

        samples_filepath = self.__samples.create_target(data_type=DataType.Test)
        samples = self.__samples.Reader.read(samples_filepath)

        graph = make_graph_from_relations_array(
            relations=self.__iter_relations(samples),
            entity_values=self.iter_column_value(samples=samples, column_value="entity_values"),
            entity_types=self.iter_column_value(samples=samples, column_value="entity_types"),
            min_links=1,
            weights=True
        )

        graph = graphs_operations(graph_A=graph, graph_B=graph, operation="SAME", min_links=0.01)

        save_force_graph(graph=graph, out_dir=dirname(target), out_filename=f"graph_force")
        save_radial_graph(graph=graph, out_dir=dirname(target), out_filename=f"graph_radial")
