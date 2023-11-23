import unittest

from tqdm import tqdm
from arelight.doc_provider import CachedFilesDocProvider
from arelight.run.utils import create_sentence_parser, iter_content


class TestCachedFiles(unittest.TestCase):

    def test(self):

        sentence_parser = create_sentence_parser(framework="nltk", language="english")
        doc_provider = CachedFilesDocProvider(
            filepaths=[
                "data/responses-d3js-backend-sample-data.csv",
                "data/rus_input_text_example.txt"
            ],
            content_provider=lambda filepath: iter_content(filepath=filepath, csv_column="text_a", csv_delimiter=","),
            content_to_sentences=sentence_parser)

        doc_ids = list(doc_provider.iter_doc_ids())
        for doc_id in tqdm(doc_ids):
            doc_provider.by_id(doc_id)


if __name__ == '__main__':
    unittest.main()
