import json
import unittest

from arelight.run.infer import create_infer_parser
from argparse_to_json import convert_parser_to_json

from arelight.run.operations import create_operations_parser


class TestArgumentsReader(unittest.TestCase):

    def test(self):
        infer_parser = create_infer_parser()
        print(json.dumps(convert_parser_to_json(infer_parser), indent=4))
        operations_parser = create_operations_parser()
        print(json.dumps(convert_parser_to_json(operations_parser), indent=4))
