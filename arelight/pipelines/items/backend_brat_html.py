from os.path import join

from arekit.common.pipeline.context import PipelineContext
from arekit.common.pipeline.items.base import BasePipelineItem

from arelight.backend.brat.ui_web import get_web_ui


class BratHtmlEmbeddingPipelineItem(BasePipelineItem):

    def apply_core(self, input_data, pipeline_ctx):
        assert(isinstance(input_data, PipelineContext))
        assert(isinstance(pipeline_ctx, PipelineContext))

        # Initialize brat-url (Optional from the pipeline context configuration).
        brat_url = pipeline_ctx.provide_or_none("brat_url")
        brat_url = "http://localhost:8001/" if brat_url is None else brat_url

        # Instantiate template.
        template = get_web_ui(coll_data=input_data.provide("coll_data"),
                              doc_data=input_data.provide("doc_data"),
                              brat_url=brat_url,
                              text=input_data.provide("text"))

        # Setup predicted result writer.
        template_fp = pipeline_ctx.provide_or_none("brat_vis_fp")
        if template_fp is None:
            exp_root = pipeline_ctx.provide_or_none("exp_root")
            template_fp = join(exp_root, "brat_output.html")

        # Save results.
        with open(template_fp, "w") as output:
            output.write(template)
