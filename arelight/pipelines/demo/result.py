from arekit.common.pipeline.context import PipelineContext

from arelight.run.utils import merge_dictionaries


class PipelineResult(PipelineContext):
    """ This structure describes the common output
        for the all items of the pipeline.
    """

    def __init__(self, extra_params=None):
        assert(isinstance(extra_params, dict) or extra_params is None)

        default_params = {
            # Inference stage -------------------------
            "iter_infer": None,
            "iter_total": None,
        }

        super(PipelineResult, self).__init__(
            merge_dictionaries(dict_iter=[default_params, {} if extra_params is None else extra_params])
        )
