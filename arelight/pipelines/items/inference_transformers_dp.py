from arekit.common.data import const
from arekit.common.data.input.providers.text.single import BaseSingleTextProvider
from arekit.common.experiment.data_type import DataType
from arekit.common.pipeline.context import PipelineContext
from arekit.common.pipeline.items.base import BasePipelineItem
from arekit.contrib.bert.input.providers.text_pair import PairTextProvider

from arelight.predict_provider import BasePredictProvider
from arelight.utils import auto_import


class TransformersDeepPavlovInferencePipelineItem(BasePipelineItem):

    def __init__(self, pretrained_bert=None, max_seq_length=128, batch_size=10):
        # Model classifier.
        self.__predict_provider = BasePredictProvider()
        self.__model = None
        self.__proc = None
        self.__max_seq_length = max_seq_length
        self.__pretrained_bert = pretrained_bert
        self.__batch_size = batch_size

    def __iter_predict_result(self, samples, batch_size):

        used_row_ids = set()

        data = {BaseSingleTextProvider.TEXT_A: [],
                PairTextProvider.TEXT_B: [],
                "row_ids": []}

        for row_ind, row in samples:

            # Considering unique rows only.
            if row[const.ID] in used_row_ids:
                continue

            data[BaseSingleTextProvider.TEXT_A].append(row[BaseSingleTextProvider.TEXT_A])
            data[PairTextProvider.TEXT_B].append(row[PairTextProvider.TEXT_B])
            data["row_ids"].append(row_ind)

            used_row_ids.add(row[const.ID])

        for i in range(0, len(data[BaseSingleTextProvider.TEXT_A]), 10):

            texts_a = data[BaseSingleTextProvider.TEXT_A][i:i + batch_size]
            texts_b = data[PairTextProvider.TEXT_B][i:i + batch_size]
            row_ids = data["row_ids"][i:i + batch_size]

            batch_features = self.__proc(texts_a=texts_a, texts_b=texts_b)

            for i, uint_label in enumerate(self.__model(batch_features)):
                yield [row_ids[i], int(uint_label)]

    def apply_core(self, input_data, pipeline_ctx):
        assert(isinstance(input_data, PipelineContext))
        assert(isinstance(pipeline_ctx, PipelineContext))

        # Fetching the batch-size from the parameters.
        labels_scaler = input_data.provide("labels_scaler")
        samples_io = input_data.provide("samples_io")
        samples_filepath = samples_io.create_target(data_type=DataType.Test)

        # If bert model has not been initialized.
        if self.__model is None:

            # Dynamic import for the deepavlov components.
            torch_preprocessor_model = auto_import(
                "deeppavlov.models.preprocessors.torch_transformers_preprocessor.TorchTransformersPreprocessor")
            torch_classifier_model = auto_import(
                "deeppavlov.models.torch_bert.torch_transformers_classifier.TorchTransformersClassifierModel")

            # Initialize bert model.
            self.__model = torch_classifier_model(pretrained_bert=self.__pretrained_bert,
                                                  n_classes=labels_scaler.LabelsCount,
                                                  bert_config_file=None,
                                                  save_path="")
            # Setup processor.
            self.__proc = torch_preprocessor_model(
                # Consider the same as pretrained BERT.
                vocab_file=self.__pretrained_bert,
                max_seq_length=self.__max_seq_length)

        iter_infer = self.__iter_predict_result(samples=samples_io.Reader.read(samples_filepath),
                                                batch_size=self.__batch_size)
        input_data.update("iter_infer", iter_infer)
