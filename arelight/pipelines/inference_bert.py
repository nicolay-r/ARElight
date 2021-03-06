from os.path import join

from arekit.common.data.storages.base import BaseRowsStorage
from arekit.common.experiment.data_type import DataType
from arekit.common.labels.scaler.base import BaseLabelScaler
from arekit.common.pipeline.context import PipelineContext
from arekit.common.pipeline.items.base import BasePipelineItem
from arekit.contrib.networks.core.predict.base_writer import BasePredictWriter
from arekit.contrib.networks.core.predict.provider import BasePredictProvider
from deeppavlov.models.bert import bert_classifier
from deeppavlov.models.preprocessors.bert_preprocessor import BertPreprocessor

from arelight.exp.exp_io import InferIOUtils


class BertInferencePipelineItem(BasePipelineItem):

    def __init__(self, bert_config_file, model_checkpoint_path, vocab_filepath,
                 data_type, predict_writer, labels_scaler, max_seq_length, do_lowercase):
        assert(isinstance(predict_writer, BasePredictWriter))
        assert(isinstance(data_type, DataType))
        assert(isinstance(labels_scaler, BaseLabelScaler))
        assert(isinstance(do_lowercase, bool))
        assert(isinstance(max_seq_length, int))

        # Model classifier.
        self.__model = bert_classifier.BertClassifierModel(
            bert_config_file=bert_config_file,
            load_path=model_checkpoint_path,
            keep_prob=1.0,
            n_classes=labels_scaler.LabelsCount,
            save_path="")

        # Setup processor.
        self.__proc = BertPreprocessor(vocab_file=vocab_filepath,
                                       do_lower_case=do_lowercase,
                                       max_seq_length=max_seq_length)

        self.__writer = predict_writer
        self.__data_type = data_type
        self.__labels_scaler = labels_scaler
        self.__predict_provider = BasePredictProvider()

    def apply_core(self, input_data, pipeline_ctx):
        assert(isinstance(input_data, InferIOUtils))
        assert(isinstance(pipeline_ctx, PipelineContext))

        def __iter_predict_result():
            samples = BaseRowsStorage.from_tsv(samples_filepath)

            used_row_ids = set()
            
            data = {"text_a": [], "text_b": [], "row_ids": []}

            for row_ind, row in samples:
                
                # Considering unique rows only.
                if row["id"] in used_row_ids:
                    continue

                data["text_a"].append(row['text_a'])
                data["text_b"].append(row['text_b'])
                data["row_ids"].append(row_ind)
                
                used_row_ids.add(row["id"])

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
