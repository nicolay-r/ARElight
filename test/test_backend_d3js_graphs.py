import os
import unittest
import utils
from os.path import join, exists
import pandas as pd

from arekit.common.pipeline.base import BasePipeline


from arelight.backend.d3js.relations_graph_builder import make_graph_from_relations_array
from arelight.backend.d3js.relations_graph_operations import graphs_operations
from arelight.backend.d3js.ui_web_force import save_force_graph
from arelight.backend.d3js.ui_web_radial import save_radial_graph
from arelight.pipelines.demo.infer_bert import demo_infer_texts_bert_pipeline
from arelight.pipelines.demo.labels.formatter import TrheeLabelsFormatter
from arelight.pipelines.demo.labels.scalers import ThreeLabelScaler
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
            relations.append([source, target, label])

        graph = make_graph_from_relations_array(
            relations=relations,
            entity_values=[item.split(',') for item in data_single_type["entity_values"]],
            entity_types=[item.split(',') for item in data_single_type["entity_types"]],
            min_links=1,
            weights=True
        )

        graph = graphs_operations(
            graph_A=graph, graph_B=graph, operation="SAME",
            min_links=0.01
        )

        if not exists(self.TEST_OUT_LOCAL_DIR):
            os.makedirs(self.TEST_OUT_LOCAL_DIR)

        save_force_graph(graph=graph, out_dir=utils.TEST_OUT_DIR, out_filename=f"graph_force_{relation_type}")
        save_radial_graph(graph=graph, out_dir=utils.TEST_OUT_DIR, out_filename=f"graph_radial_{relation_type}")

        # Launch server to checkout the results.
        os.system(f"cd {self.TEST_OUT_LOCAL_DIR} && python -m http.server 8000")

    def test_pipeline(self):

        # TIP: you need to launch test_pipeline_sample.py first!

        ppl = demo_infer_texts_bert_pipeline(sampling_engines=None,
                                             backend_engines="d3js_graphs")

        pipeline = BasePipeline(ppl)
        ppl_result = PipelineResult()
        ppl_result.update("predict_filepath", value=join(utils.TEST_OUT_DIR, "predict.tsv.gz"))
        ppl_result.update("labels_formatter", value=TrheeLabelsFormatter())
        ppl_result.update("labels_scaler", value=ThreeLabelScaler())

        pipeline.run(input_data=ppl_result,
                     params_dict={
                        "backend_template": self.TEST_OUT_LOCAL_DIR + "out"
                     })
