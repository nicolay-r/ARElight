import json
import unittest

from arelight.backend.d3js.relations_graph_operations import OP_UNION, OP_INTERSECTION, OP_DIFFERENCE
from arelight.run.infer import create_infer_parser
from argparse_to_json import convert_parser_to_json

from arelight.run.operations import create_operations_parser


class TestArgumentsReader(unittest.TestCase):

    @staticmethod
    def extract(parser):
        values = vars(parser.parse_args())
        json_data = convert_parser_to_json(parser)
        for k, v in values.items():
            json_data["schema"][k]["default"] = v
        return json_data

    def test(self):
        infer_parser = create_infer_parser()
        infer_schema = self.extract(infer_parser)
        print(json.dumps(infer_schema, indent=4))
        operations_parser = create_operations_parser([OP_UNION, OP_INTERSECTION, OP_DIFFERENCE])
        operations_schema = self.extract(operations_parser)
        print(json.dumps(operations_schema, indent=4))
