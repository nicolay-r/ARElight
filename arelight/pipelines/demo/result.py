from arekit.common.pipeline.context import PipelineContext
from arekit.contrib.utils.data.readers.csv_pd import PandasCsvReader

from arelight.pipelines.demo.labels.formatter import TrheeLabelsFormatter
from arelight.pipelines.demo.labels.scalers import ThreeLabelScaler
from arelight.predict_writer_csv import TsvPredictWriter
from arelight.run.utils import merge_dictionaries


class PipelineResult(PipelineContext):
    """ This structure describes the common output
        for the all items of the pipeline.
    """

    def __init__(self, extra_params):
        assert(isinstance(extra_params, dict))

        default_params = {
            # Serialization parameters ----------------
            "labels_formatter": TrheeLabelsFormatter(),
            "labels_scaler": ThreeLabelScaler(),
            # Inference stage -------------------------
            "iter_infer": None,
            # Inference stage -------------------------
            "predict_filepath": None,
            "predict_writer": TsvPredictWriter(),  # The way we save the predicted results.
            "predict_reader": PandasCsvReader(compression='infer'),  # The way we can read the predicted results.
        }

        super(PipelineResult, self).__init__(
            merge_dictionaries(dict_iter=[default_params, extra_params])
        )
