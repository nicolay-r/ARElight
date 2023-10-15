import json
import os
import torch

from arekit.common.experiment.data_type import DataType
from arekit.common.pipeline.context import PipelineContext
from arekit.common.pipeline.items.base import BasePipelineItem

from opennre.encoder import BERTEntityEncoder, BERTEncoder
from opennre.framework import SentenceRELoader
from opennre.model import SoftmaxNN

from arelight.pipelines.items.utils import try_download_predefined_checkpoint
from arelight.utils import get_default_download_dir


class BertOpenNREInferencePipelineItem(BasePipelineItem):

    def __init__(self, pretrained_bert=None, checkpoint_path=None, device_type='cpu',
                 max_seq_length=128, pooler='cls', batch_size=10, tokenizers_parallelism=True):
        assert(isinstance(tokenizers_parallelism, bool))

        self.__model = None
        self.__pretrained_bert = pretrained_bert
        self.__checkpoint_path = checkpoint_path
        self.__device_type = device_type
        self.__max_seq_length = max_seq_length
        self.__pooler = pooler
        self.__batch_size = batch_size

        # Huggingface/Tokenizers compatibility.
        os.environ['TOKENIZERS_PARALLELISM'] = str(tokenizers_parallelism).lower()

    @staticmethod
    def load_bert_sentence_encoder(pooler, max_length, pretrain_path, mask_entity):
        """ We support two type of models: `cls` based and `entity` based.
        """
        if pooler == 'entity':
            return BERTEntityEncoder(
                max_length=max_length,
                pretrain_path=pretrain_path,
                mask_entity=mask_entity
            )
        elif pooler == 'cls':
            return BERTEncoder(
                max_length=max_length,
                pretrain_path=pretrain_path,
                mask_entity=mask_entity
            )
        else:
            raise NotImplementedError

    @staticmethod
    def init_bert_model(pretrain_path, rel2id, ckpt_path, device_type, dir_to_donwload=None,
                        pooler='cls', max_length=128, mask_entity=True):
        """ This is a main and core method for inference based on OpenNRE framework.
        """
        # Check predefined checkpoints for local downloading.
        predefined_pretrain_path, predefined_ckpt_path = try_download_predefined_checkpoint(
            checkpoint=ckpt_path, dir_to_download=dir_to_donwload)

        # Update checkpoint and pretrain paths with the predefined.
        ckpt_path = predefined_ckpt_path if predefined_ckpt_path is not None else ckpt_path
        pretrain_path = predefined_pretrain_path if predefined_pretrain_path is not None else pretrain_path

        # Load original model.
        bert_encoder = BertOpenNREInferencePipelineItem.load_bert_sentence_encoder(
            pooler=pooler, mask_entity=mask_entity, max_length=max_length, pretrain_path=pretrain_path)
        # Load checkpoint.
        model = SoftmaxNN(bert_encoder, len(rel2id), rel2id)
        model.load_state_dict(torch.load(ckpt_path, map_location=torch.device(device_type))['state_dict'])
        return model

    @staticmethod
    def extract_ids(data_file):
        with open(data_file) as input_file:
            for line_str in input_file.readlines():
                data = json.loads(line_str)
                yield data["id_orig"]

    @staticmethod
    def iter_results(parallel_model, eval_loader, data_ids):
        l_ind = 0
        with torch.no_grad():
            for iter, data in enumerate(eval_loader):
                if torch.cuda.is_available():
                    for i in range(len(data)):
                        try:
                            data[i] = data[i].cuda()
                        except:
                            pass

                args = data[1:]
                logits = parallel_model(*args)
                score, pred = logits.max(-1)  # (B)

                # Save result
                batch_size = pred.size(0)
                for i in range(batch_size):
                    yield data_ids[l_ind], pred[i].item()
                    l_ind += 1

    def __iter_predict_result(self, samples_filepath, batch_size):
        # Compose evaluator.
        sentence_eval = SentenceRELoader(path=samples_filepath,
                                         rel2id=self.__model.rel2id,
                                         tokenizer=self.__model.sentence_encoder.tokenize,
                                         batch_size=batch_size,
                                         shuffle=False)

        # Iter output results.
        results_it = self.iter_results(parallel_model=torch.nn.DataParallel(self.__model),
                                       data_ids=list(self.extract_ids(samples_filepath)),
                                       eval_loader=sentence_eval)

        total = len(sentence_eval.dataset)

        return results_it, total

    def apply_core(self, input_data, pipeline_ctx):
        assert(isinstance(input_data, PipelineContext))

        # Fetching the input data.
        labels_scaler = input_data.provide("labels_scaler")

        # Try to obrain from the specific input variable.
        samples_filepath = input_data.provide_or_none("opennre_samples_filepath")
        if samples_filepath is None:
            samples_io = input_data.provide("samples_io")
            samples_filepath = samples_io.create_target(data_type=DataType.Test)

        # We compose specific mapping required by opennre to perform labels mapping.
        rel2id = {}
        for label in labels_scaler.ordered_suppoted_labels():
            uint_label = labels_scaler.label_to_uint(label)
            rel2id[str(uint_label)] = uint_label

        # Initialize model if the latter has not been yet.
        if self.__model is None:

            ckpt_dir = input_data.provide_or_none("opennre_ckpt_cache_dir")

            self.__model = self.init_bert_model(
                pretrain_path=self.__pretrained_bert,
                ckpt_path=self.__checkpoint_path,
                device_type=self.__device_type,
                max_length=self.__max_seq_length,
                pooler=self.__pooler,
                rel2id=rel2id,
                mask_entity=True,
                dir_to_donwload=get_default_download_dir() if ckpt_dir is None else ckpt_dir)

        iter_infer, total = self.__iter_predict_result(samples_filepath=samples_filepath, batch_size=self.__batch_size)
        input_data.update("iter_infer", iter_infer)
        input_data.update("iter_total", total)
