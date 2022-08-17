import json
from os.path import join

from arekit.common.pipeline.context import PipelineContext
from arekit.common.pipeline.items.base import BasePipelineItem


class BratHtmlEmbeddingPipelineItem(BasePipelineItem):

    def __init__(self, brat_url="http://localhost:8001/"):
        self.__brat_url = brat_url

    def apply_core(self, input_data, pipeline_ctx):
        assert(isinstance(input_data, dict))
        assert(isinstance(pipeline_ctx, PipelineContext))

        # Loading template file.
        template_filepath = pipeline_ctx.provide_or_none("template_filepath")
        with open(template_filepath, "r") as templateFile:
            template = templateFile.read()

        # Replace template placeholders.
        template = template.replace("$____COL_DATA_SEM____", json.dumps(input_data["coll_data"]))
        template = template.replace("$____DOC_DATA_SEM____", json.dumps(input_data["doc_data"]))
        template = template.replace("$____BRAT_URL____", self.__brat_url)
        template = template.replace("$____TEXT____", input_data["text"])

        # Setup predicted result writer.
        template_fp = pipeline_ctx.provide_or_none("brat_vis_fp")
        if template_fp is None:
            exp_root = pipeline_ctx.provide_or_none("exp_root")
            template_fp = join(exp_root, "brat_output.html")

        # Save results.
        with open(template_fp, "w") as output:
            output.write(template)

        return template
