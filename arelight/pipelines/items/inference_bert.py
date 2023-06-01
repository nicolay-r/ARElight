from os.path import join, dirname

from arekit.common.data import const
from arekit.common.data.input.providers.text.single import BaseSingleTextProvider
from arekit.common.experiment.data_type import DataType
from arekit.common.pipeline.context import PipelineContext
from arekit.common.pipeline.items.base import BasePipelineItem
from arekit.contrib.bert.input.providers.text_pair import PairTextProvider
from arekit.contrib.utils.io_utils.samples import SamplesIO

from arelight.predict_provider import BasePredictProvider
from arelight.predict_writer import BasePredictWriter

from deeppavlov.models.bert import bert_classifier
from deeppavlov.models.preprocessors.bert_preprocessor import BertPreprocessor


class BertInferencePipelineItem(BasePipelineItem):

    def __init__(self, bert_config_file, model_checkpoint_path, vocab_filepath, samples_io,
                 data_type, predict_writer, labels_count, max_seq_length, do_lowercase,
                 batch_size=10):
        assert(isinstance(predict_writer, BasePredictWriter))
        assert(isinstance(data_type, DataType))
        assert(isinstance(labels_count, int))
        assert(isinstance(do_lowercase, bool))
        assert(isinstance(max_seq_length, int))
        assert(isinstance(samples_io, SamplesIO))

        # Model classifier.
        self.__model = bert_classifier.BertClassifierModel(
            bert_config_file=bert_config_file,
            load_path=model_checkpoint_path,
            keep_prob=1.0,
            n_classes=labels_count,
            save_path="")

        # Setup processor.
        self.__proc = BertPreprocessor(vocab_file=vocab_filepath,
                                       do_lower_case=do_lowercase,
                                       max_seq_length=max_seq_length)

        self.__writer = predict_writer
        self.__data_type = data_type
        self.__labels_count = labels_count
        self.__predict_provider = BasePredictProvider()
        self.__samples_io = samples_io
        self.__batch_size = batch_size

    def apply_core(self, input_data, pipeline_ctx):
        assert(isinstance(pipeline_ctx, PipelineContext))

        def __iter_predict_result():
            samples = self.__samples_io.Reader.read(samples_filepath)

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

                texts_a = data[BaseSingleTextProvider.TEXT_A][i:i + self.__batch_size]
                texts_b = data[PairTextProvider.TEXT_B][i:i + self.__batch_size]
                row_ids = data["row_ids"][i:i + self.__batch_size]

                batch_features = self.__proc(texts_a=texts_a, texts_b=texts_b)

                for i, uint_label in enumerate(self.__model(batch_features)):
                    yield [row_ids[i], int(uint_label)]

        # Fetch other required in furter information from input_data.
        samples_filepath = self.__samples_io.create_target(
            data_type=self.__data_type,
            data_folding=pipeline_ctx.provide("data_folding"))

        # Setup predicted result writer.
        tgt = pipeline_ctx.provide_or_none("predict_fp")
        if tgt is None:
            tgt = join(dirname(samples_filepath), "predict.tsv.gz")

        # Setup target filepath.
        self.__writer.set_target(tgt)

        # Update for further pipeline items.
        pipeline_ctx.update("predict_fp", tgt)

        # Gathering the content
        title, contents_it = self.__predict_provider.provide(
            sample_id_with_uint_labels_iter=__iter_predict_result(),
            labels_count=self.__labels_count)

        with self.__writer:
            self.__writer.write(title=title, contents_it=contents_it)

        return self.__samples_io
