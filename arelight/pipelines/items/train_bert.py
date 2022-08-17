from arekit.common.data import const
from arekit.common.data.storages.base import BaseRowsStorage
from arekit.common.pipeline.context import PipelineContext
from arekit.common.pipeline.items.base import BasePipelineItem
from arekit.common.synonyms.base import SynonymsCollection
from deeppavlov.models.bert import bert_classifier
from deeppavlov.models.preprocessors.bert_preprocessor import BertPreprocessor
from tqdm import tqdm


class BertFinetunePipelineItem(BasePipelineItem):

    def __init__(self, bert_config_file, model_checkpoint_path, do_lowercase,
                 learning_rate, vocab_filepath, max_seq_length, save_path):
        assert(isinstance(bert_config_file, str))
        assert(isinstance(model_checkpoint_path, str))

        # Model classifier.
        self.__model = bert_classifier.BertClassifierModel(
            bert_config_file=bert_config_file,
            load_path=model_checkpoint_path,
            keep_prob=0.1,
            n_classes=3,
            save_path=save_path,
            learning_rate=learning_rate)

        # Setup processor.
        self.__proc = BertPreprocessor(vocab_file=vocab_filepath,
                                       do_lower_case=do_lowercase,
                                       max_seq_length=max_seq_length)

    @staticmethod
    def get_synonym_group_index(synonyms, value):
        assert(isinstance(synonyms, SynonymsCollection))
        if not synonyms.contains_synonym_value(value):
            synonyms.add_synonym_value(value)
        return synonyms.get_synonym_group_index(value)

    def apply_core(self, input_data, pipeline_ctx):
        assert(isinstance(input_data, str))
        assert(isinstance(pipeline_ctx, PipelineContext))

        def __iter_batches(s, batch_size):
            assert(isinstance(s, BaseRowsStorage))

            data = {"text_a": [], "text_b": [], "label": []}

            # NOTE: it is important to iter shuffled data!
            for row_ind, row in s.iter_shuffled():
                data["text_a"].append(row['text_a'])
                data["text_b"].append(row['text_b'])
                data["label"].append(row[const.LABEL])

            for i in range(0, len(data["text_a"]), batch_size):

                texts_a = data["text_a"][i:i + batch_size]
                texts_b = data["text_b"][i:i + batch_size]
                labels = data["label"][i:i + batch_size]

                batch_features = self.__proc(texts_a=texts_a, texts_b=texts_b)

                yield batch_features, labels

        # Reading pipeline parameters.
        epochs_count = pipeline_ctx.provide("epochs_count")
        batch_size = pipeline_ctx.provide("batch_size")
        samples = BaseRowsStorage.from_tsv(input_data)

        for e in range(epochs_count):

            it = __iter_batches(samples, batch_size)
            batches = len(samples.DataFrame) / batch_size

            total_loss = 0
            pbar = tqdm(it, total=batches, desc="Epoch: {}".format(e), unit='batches')
            for batch_index, payload in enumerate(pbar):
                features, y = payload
                d = self.__model.train_on_batch(features=features, y=y)
                total_loss += d["loss"]
                pbar.set_postfix({
                    "avg-loss": total_loss/(batch_index+1)
                })

        # Save the result model.
        self.__model.save()
