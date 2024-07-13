import os
import unittest
import utils
from os.path import join, exists
import pandas as pd

from arekit.common.pipeline.base import BasePipelineLauncher
from arekit.contrib.utils.data.readers.jsonl import JsonlReader
from arekit.contrib.utils.data.readers.csv_pd import PandasCsvReader

from arelight.arekit.samples_io import CustomSamplesIO
from arelight.backend.d3js.relations_graph_builder import make_graph_from_relations_array
from arelight.backend.d3js.relations_graph_operations import graphs_operations
from arelight.backend.d3js.utils_graph import save_graph
from arelight.pipelines.demo.infer_bert import demo_infer_texts_bert_pipeline
from arelight.pipelines.demo.labels.formatter import CustomLabelsFormatter
from arelight.pipelines.demo.labels.scalers import CustomLabelScaler
from arelight.pipelines.demo.result import PipelineResult


class TestBackendD3JS(unittest.TestCase):

    TEST_OUT_LOCAL_DIR = join(utils.TEST_OUT_DIR, "d3js_test/")

    def test(self):

        # Reading source file.
        data = pd.read_csv(join(utils.TEST_DATA_DIR, "responses-d3js-backend-sample-data.csv"))

        # Value based visualization.
        relation_type = "WORKS_AS"
        column_name = "relations_pretty_value"

        data_single_type = data[data["relation_type"] == relation_type]

        # Formatting the content of file.
        relations = []
        for x in data_single_type[column_name]:
            rel, label = x.split(';')
            source, target = rel.split('->')

            # Remove brackets in original annotation.
            source = source.replace("(", "")
            label = label.replace(")", "")

            relations.append([source, target, label])

        graph = make_graph_from_relations_array(
            graph_name="UNKNOWN_GRAPH_NAME",
            relations=relations[10:],
            entity_values=[item.split(',') for item in data_single_type["entity_values"]],
            entity_types=[item.split(',') for item in data_single_type["entity_types"]],
            min_links=1,
            weights=True
        )

        graph2 = make_graph_from_relations_array(
            graph_name="UNKNOWN_GRAPH_NAME",
            relations=relations[:10],
            entity_values=[item.split(',') for item in data_single_type["entity_values"]],
            entity_types=[item.split(',') for item in data_single_type["entity_types"]],
            min_links=1,
            weights=True
        )

        graph = graphs_operations(graph_A=graph2, graph_B=graph, operation="DIFFERENCE", weights=False)
        print(graph)

        if not exists(self.TEST_OUT_LOCAL_DIR):
            os.makedirs(self.TEST_OUT_LOCAL_DIR)

        save_graph(graph=graph, out_dir=utils.TEST_OUT_DIR, out_filename=f"./force/graph_{relation_type}", convert_to_radial=False)
        save_graph(graph=graph, out_dir=utils.TEST_OUT_DIR, out_filename=f"./radial/graph_{relation_type}", convert_to_radial=True)

        # Launch server to checkout the results.
        os.system(f"cd {utils.TEST_OUT_DIR} && python -m http.server 8001")

    def test_pipeline(self):

        # TIP: you need to launch test_pipeline_sample.py first!
        ppl = demo_infer_texts_bert_pipeline(
            sampling_engines=None,
            backend_engines={
                "d3js_graphs": {
                    "graph_min_links": 0.1,
                    "graph_a_labels": None,
                    "weights": True,
                }
            })

        target_func = lambda data_type: join(utils.TEST_OUT_DIR, "-".join(["sample", data_type.name.lower()]))
        samples_io = CustomSamplesIO(create_target_func=target_func, reader=JsonlReader())

        ppl_result = PipelineResult(extra_params={
            "samples_io": samples_io,
            "labels_scaler": CustomLabelScaler(),
            "d3js_graph_output_dir": utils.TEST_OUT_DIR,
            "predict_reader": PandasCsvReader(compression='infer'),
        })
        ppl_result.update("predict_filepath", value=join(utils.TEST_OUT_DIR, "predict.tsv.gz"))
        ppl_result.update("labels_formatter", value=CustomLabelsFormatter())

        BasePipelineLauncher.run(pipeline=ppl, pipeline_ctx=ppl_result, src_key="samples_io")
