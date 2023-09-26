#!/usr/bin/python3

import cgi
import cgitb
from os.path import realpath, dirname, join

from arekit.common.pipeline.base import BasePipeline
from arekit.contrib.utils.data.readers.csv_pd import PandasCsvReader
from arekit.contrib.utils.io_utils.samples import SamplesIO

from arelight.pipelines.demo.infer_bert import demo_infer_texts_bert_pipeline
from arelight.pipelines.demo.result import PipelineResult

cgitb.enable()

# We consider OpenNRE engine for inference texts written in Russian.
INFER_ENGINES = {
    "opennre": {
        "pretrained_bert": "DeepPavlov/rubert-base-cased",
        "checkpoint_path": "ra4-rsr1_DeepPavlov-rubert-base-cased_cls.pth.tar",
        "device_type": "cpu",
        "max_seq_length": 128
    }
}

# Setup folder for output.
OUTPUT_DIR = join(dirname(realpath(__file__)), "./output")


def do_infer(text):
    assert(isinstance(text, str) or isinstance(text, list) or text is None)

    # Forming pipeline for inference only.
    pipeline = demo_infer_texts_bert_pipeline(
        sampling_engines=None,
        infer_engines=INFER_ENGINES,
        backend_engines=["d3js_graphs"])

    ppl = BasePipeline(pipeline)

    ppl_result = PipelineResult(extra_params={
        # Just for reading samples.
        "samples_io": SamplesIO(target_dir=OUTPUT_DIR, reader=PandasCsvReader(sep=',', compression=None), prefix="samples"),
        # Since OpenNRE has it's own format, there is a need to provide path to it.
        "opennre_samples_filepath": join(OUTPUT_DIR, "samples-test-0.jsonl"),
        # Output directory.
        "d3js_graph_output_dir": OUTPUT_DIR,
        # Cancelling saving option.
        "d3js_graph_do_save": False
    })

    ppl.run(input_data=ppl_result)

    # Extract from the pipeline.
    return ppl_result.provide("d3js_graph_radial_html_template")


def main():
    form = cgi.FieldStorage()

    # Fetching the input data (assuming textarea's name attribute is 'data')
    # input_text = form.getvalue('data', 'No data provided')

    d3js_html_content = do_infer(text=None)

    print("Content-type: text/plain\n")
    print(d3js_html_content)


if __name__ == "__main__":
    main()
