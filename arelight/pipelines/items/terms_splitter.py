from arekit.common.pipeline.context import PipelineContext
from arekit.common.pipeline.items.base import BasePipelineItem
from arekit.common.utils import split_by_whitespaces


class CustomTermsSplitterPipelineItem(BasePipelineItem):

    def apply_core(self, input_data, pipeline_ctx):
        assert(isinstance(pipeline_ctx, PipelineContext))
        terms = []
        for e in input_data:
            if isinstance(e, str):
                terms.extend(split_by_whitespaces(e))
            else:
                terms.append(e)
        return terms
