from arekit.common.pipeline.context import PipelineContext
from arekit.contrib.utils.data.readers.csv_pd import PandasCsvReader


class PipelineResult(PipelineContext):
    """ This structure describes the common output
        for the all items of the pipeline.
    """

    def __init__(self):
        super(PipelineResult, self).__init__(
            d={
                # Inference stage -------------------------
                "predict_filepaths": [],
                "predict_labels_formatter": None,
                "predict_labels_scaler": None,
                "predict_reader": PandasCsvReader(compression='infer'),  # The way we can read the predicted results.
                "predict_writer": None,                                  # The way we save the predicted results.
        })
