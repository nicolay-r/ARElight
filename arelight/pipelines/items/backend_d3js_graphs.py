import logging
import os

from arekit.common.data import const
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


logger = logging.getLogger(__name__)


class D3jsGraphsBackendPipelineItem(BasePipelineItem):

    # List of the supported graphs with their UI functions.
    graphs_web_ui = {
        "radial": get_radial_web_ui,
        "force": get_force_web_ui,
    }

    def __init__(self, operation_type="SAME", graph_min_links=1, op_min_links=0.01, ui_output=None,
                 graph_a_labels=None, graph_b_labels=None, weights=True, launch_server=False):
        assert(isinstance(operation_type, str) and operation_type in ["SAME", "DIFF", "PLUS", "MINUS"])
        self.__operation_type = operation_type
        self.__graph_min_links = graph_min_links
        self.__op_min_links = op_min_links
        self.__ui_types_output = list(self.graphs_web_ui.keys()) if ui_output is None else ui_output

        # Setup filters for the A and B graphs for further operations application.
        self.__graph_a_filter = lambda _: True if graph_a_labels is None else lambda label: label in graph_a_labels
        self.__graph_b_filter = lambda _: True if graph_b_labels is None else lambda label: label in graph_b_labels

        # Considering weights for grapsh.
        self.__graph_weights = weights
        self.__launch_server = launch_server

    @staticmethod
    def __iter_relations(samples, labels, labels_filter_func=None):
        assert(isinstance(samples, BaseRowsStorage))
        assert(isinstance(labels, list))
        assert(callable(labels_filter_func) or labels_filter_func is None)

        for ind, row_data in enumerate(samples):
            _, sample_row = row_data
            parsed_row = ParsedSampleRow(sample_row)
            label = labels[ind]

            # Optional filtering
            if labels_filter_func is not None and not labels_filter_func(label):
                continue

            source = ParsedSampleRowExtraService.calc("SourceValue", parsed_row=parsed_row)
            target = ParsedSampleRowExtraService.calc("TargetValue", parsed_row=parsed_row)
            yield [source, target, label]

    def iter_column_value(self, samples, column_value):
        for _, sample_row in samples:
            parsed_row = ParsedSampleRow({column_value: sample_row[column_value]})
            yield parsed_row[column_value]

    def apply_core(self, input_data, pipeline_ctx):
        assert(isinstance(input_data, PipelineContext))

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
        labels_to_str = {str(labels_scaler.label_to_uint(label)): labels_fmt.label_to_str(label)
                         for label in labels_scaler.ordered_suppoted_labels()}
        labels = list(iter_predicted_labels(predict_data=predict_storage, label_to_str=labels_to_str, keep_ind=False))

        graph_a = make_graph_from_relations_array(
            relations=self.__iter_relations(samples, labels, labels_filter_func=self.__graph_a_filter),
            entity_values=self.iter_column_value(samples=samples, column_value=const.ENTITY_VALUES),
            entity_types=self.iter_column_value(samples=samples, column_value=const.ENTITY_TYPES),
            min_links=self.__graph_min_links,
            weights=self.__graph_weights)

        graph_b = make_graph_from_relations_array(
            relations=self.__iter_relations(samples, labels, labels_filter_func=self.__graph_b_filter),
            entity_values=self.iter_column_value(samples=samples, column_value=const.ENTITY_VALUES),
            entity_types=self.iter_column_value(samples=samples, column_value=const.ENTITY_TYPES),
            min_links=self.__graph_min_links,
            weights=self.__graph_weights)

        # Calculate the result graph.
        graph = graphs_operations(graph_A=graph_a, graph_B=graph_b,
                                  operation=self.__operation_type,
                                  min_links=self.__op_min_links)

        # Setups whether we would like to save the result graph template as a separate HTML file.
        do_save = input_data.provide("d3js_graph_do_save")

        for graph_ui_type, ui_func in self.graphs_web_ui.items():

            # Consider only those that predefined in options.
            if graph_ui_type not in self.__ui_types_output:
                continue

            html_content = save_graph(graph=graph, out_dir=target_dir, ui_func=ui_func,
                                      out_filename=f"graph_{graph_ui_type}",
                                      convert_to_radial=True if "radial" in graph_ui_type else False,
                                      do_save_template=do_save)

            input_data.update(f"d3js_graph_{graph_ui_type}_html_template", html_content)

        launch_server = input_data.provide_or_none("d3js_graph_launch_server")

        # Launch server to checkout the results (Optionally)
        if launch_server is not None and launch_server is True:
            cmd = f"cd {target_dir} && python -m http.server 8000"
            print("Launching WEB server for watching the results")
            logger.info(cmd)
            os.system(cmd)
