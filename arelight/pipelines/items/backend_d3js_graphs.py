from os.path import dirname

from arekit.common.data.storages.base import BaseRowsStorage
from arekit.common.experiment.data_type import DataType
from arekit.common.labels.scaler.base import BaseLabelScaler
from arekit.common.labels.str_fmt import StringLabelsFormatter
from arekit.common.pipeline.context import PipelineContext
from arekit.common.pipeline.items.base import BasePipelineItem
from arekit.contrib.networks.input.rows_parser import ParsedSampleRow

from arelight.arekit.parse_predict import iter_predicted_labels
from arelight.arekit.parsed_row_service import ParsedSampleRowExtraService
from arelight.backend.d3js.relations_graph_builder import make_graph_from_relations_array
from arelight.backend.d3js.relations_graph_operations import graphs_operations
from arelight.backend.d3js.ui_web_force import get_force_web_ui
from arelight.backend.d3js.ui_web_radial import get_radial_web_ui
from arelight.backend.d3js.utils_graph import save_graph


class D3jsGraphsBackendPipelineItem(BasePipelineItem):

    @staticmethod
    def __iter_relations(samples, labels):
        assert(isinstance(samples, BaseRowsStorage))
        assert(isinstance(labels, list))

        for ind, row_data in enumerate(samples):
            _, sample_row = row_data
            parsed_row = ParsedSampleRow(sample_row)
            source = ParsedSampleRowExtraService.calc("SourceValue", parsed_row=parsed_row)
            target = ParsedSampleRowExtraService.calc("TargetValue", parsed_row=parsed_row)
            yield [source, target, labels[ind]]

    def iter_column_value(self, samples, column_value):
        for _, sample_row in samples:
            parsed_row = ParsedSampleRow({column_value: sample_row[column_value]})
            yield parsed_row[column_value]

    def apply_core(self, input_data, pipeline_ctx):
        assert(isinstance(input_data, PipelineContext))
        assert(isinstance(pipeline_ctx, PipelineContext))

        target_dir = input_data.provide("d3js_graph_output_dir")
        predict_filepath = input_data.provide("predict_filepath")
        result_reader = input_data.provide("predict_reader")
        labels_fmt = input_data.provide("labels_formatter")
        assert(isinstance(labels_fmt, StringLabelsFormatter))
        labels_scaler = input_data.provide("labels_scaler")
        assert(isinstance(labels_scaler, BaseLabelScaler))
        predict_storage = result_reader.read(predict_filepath)
        assert(isinstance(predict_storage, BaseRowsStorage))

        # Reading samples.
        samples_io = input_data.provide("samples_io")
        samples_filepath = samples_io.create_target(data_type=DataType.Test)
        samples = samples_io.Reader.read(samples_filepath)

        # Reading labels.
        labels_to_rel = {str(labels_scaler.label_to_uint(label)): labels_fmt.label_to_str(label)
                         for label in labels_scaler.ordered_suppoted_labels()}
        labels = list(iter_predicted_labels(predict_data=predict_storage, label_to_rel=labels_to_rel, keep_ind=False))

        graph = make_graph_from_relations_array(
            relations=self.__iter_relations(samples, labels),
            entity_values=self.iter_column_value(samples=samples, column_value="entity_values"),
            entity_types=self.iter_column_value(samples=samples, column_value="entity_types"),
            min_links=1,
            weights=True
        )

        graph = graphs_operations(graph_A=graph, graph_B=graph, operation="SAME", min_links=0.01)

        do_save = input_data.provide_or_none("d3js_graph_do_save")

        html_content_force = save_graph(graph=graph, out_dir=target_dir, ui_func=get_force_web_ui,
                                        out_filename=f"graph_force", convert_to_radial=False,
                                        do_save_template=do_save if do_save is not None else True)

        html_content_radial = save_graph(graph=graph, out_dir=target_dir, ui_func=get_radial_web_ui,
                                         out_filename=f"graph_radial", convert_to_radial=True,
                                         do_save_template=do_save if do_save is not None else True)

        input_data.update("d3js_graph_radial_html_template", html_content_radial)
        input_data.update("d3js_graph_force_html_template", html_content_force)
