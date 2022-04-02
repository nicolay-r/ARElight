from os.path import join

from arekit.common.data.storages.base import BaseRowsStorage
from arekit.common.experiment.data_type import DataType
from arekit.common.pipeline.context import PipelineContext
from arekit.common.pipeline.items.base import BasePipelineItem
from arekit.contrib.networks.core.predict.base_writer import BasePredictWriter
from arekit.contrib.networks.core.predict.provider import BasePredictProvider
from deeppavlov.models.bert import bert_classifier
from deeppavlov.models.preprocessors.bert_preprocessor import BertPreprocessor

from exp.exp_io import InferIOUtils


class BertInferencePipelineItem(BasePipelineItem):

    def __init__(self, data_type, predict_writer, labels_scaler):
        assert(isinstance(predict_writer, BasePredictWriter))
        assert(isinstance(data_type, DataType))

        # Model classifier.
        self.__model = bert_classifier.BertClassifierModel(
            bert_config_file=join('models', "ra-20-srubert-large-neut-nli-pretrained-3l/bert_config.json"),
            pretrained_bert=join('models', "ra-20-srubert-large-neut-nli-pretrained-3l/model.ckpt-30238"),
            attention_probs_keep_prob=1.0,
            hidden_keep_prob=1.0,
            keep_prob=1.0,
            n_classes=3,
            save_path="")

        # Setup processor.
        self.__proc = BertPreprocessor(
            vocab_file=join('models', "ra-20-srubert-large-neut-nli-pretrained-3l/vocab.txt"),
            do_lower_case=False,
            max_seq_length=128)

        self.__writer = predict_writer
        self.__data_type = data_type
        self.__labels_scaler = labels_scaler
        self.__predict_provider = BasePredictProvider()

    def apply_core(self, input_data, pipeline_ctx):
        assert(isinstance(input_data, InferIOUtils))
        assert(isinstance(pipeline_ctx, PipelineContext))

        def __iter_predict_result():
            samples = BaseRowsStorage.from_tsv(samples_filepath)

            data = {"text_a": [], "text_b": [], "row_ids": []}

            for row_ind, row in samples:
                data["text_a"].append(row['text_a'])
                data["text_b"].append(row['text_b'])
                data["row_ids"].append(row_ind)

            batch_size = 10

            for i in range(0, len(data["text_a"]), 10):

                texts_a = data["text_a"][i:i + batch_size]
                texts_b = data["text_b"][i:i + batch_size]
                row_ids = data["row_ids"][i:i + batch_size]

                batch_features = self.__proc(texts_a=texts_a, texts_b=texts_b)

                for i, uint_label in enumerate(self.__model(batch_features)):
                    yield [row_ids[i], int(uint_label)]

        # Setup predicted result writer.
        tgt = pipeline_ctx.provide_or_none("predict_fp")
        if tgt is None:
            exp_root = join(input_data._get_experiment_sources_dir(),
                            input_data.get_experiment_folder_name())
            tgt = join(exp_root, "predict.tsv.gz")

        # Setup target filepath.
        self.__writer.set_target(tgt)

        # Fetch other required in furter information from input_data.
        samples_filepath = input_data.create_samples_writer_target(self.__data_type)

        # Update for further pipeline items.
        pipeline_ctx.update("predict_fp", tgt)

        # Gathering the content
        title, contents_it = self.__predict_provider.provide(
            sample_id_with_uint_labels_iter=__iter_predict_result(),
            labels_scaler=self.__labels_scaler)

        with self.__writer:
            self.__writer.write(title=title, contents_it=contents_it)
