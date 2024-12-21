import os
from os.path import exists, join

import logging
import torch

from arekit.common.experiment.data_type import DataType
from arekit.common.pipeline.items.base import BasePipelineItem

from opennre.encoder import BERTEntityEncoder, BERTEncoder
from opennre.model import SoftmaxNN

from arelight.third_party.torch import sentence_re_loader
from arelight.utils import get_default_download_dir, download

logger = logging.getLogger(__name__)


class BertOpenNREInferencePipelineItem(BasePipelineItem):

    def __init__(self, pretrained_bert=None, checkpoint_path=None, device_type='cpu',
                 max_seq_length=128, pooler='cls', batch_size=10, tokenizers_parallelism=True,
                 table_name="contents", task_kwargs=None, predefined_ckpts=None, logger=None,
                 data_loader_num_workers=0, **kwargs):
        """
        NOTE: data_loader_num_workers has set to 0 to cope with the following issue #147:
        https://github.com/nicolay-r/ARElight/issues/147
        where the most similar
        """
        assert(isinstance(tokenizers_parallelism, bool))
        super(BertOpenNREInferencePipelineItem, self).__init__(**kwargs)

        self.__model = None
        self.__pretrained_bert = pretrained_bert
        self.__checkpoint_path = checkpoint_path
        self.__device_type = device_type
        self.__max_seq_length = max_seq_length
        self.__pooler = pooler
        self.__batch_size = batch_size
        self.__predefined_ckpts = {} if predefined_ckpts is None else predefined_ckpts
        self.__task_kwargs = task_kwargs
        self.__table_name = table_name
        self.__logger = logger
        self.__data_loader_num_workers = data_loader_num_workers

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
    def scaler_to_rel2id(labels_scaler):
        rel2id = {}
        for label in labels_scaler.ordered_suppoted_labels():
            uint_label = labels_scaler.label_to_uint(label)
            rel2id[str(uint_label)] = uint_label
        return rel2id

    @staticmethod
    def try_download_predefined_checkpoint(checkpoint, predefined, dir_to_download, logger=None):
        """ This is for the simplicity of using the framework straightaway.
        """
        assert (isinstance(checkpoint, str))
        assert (isinstance(dir_to_download, str))

        if checkpoint in predefined:
            data = predefined[checkpoint]
            target_checkpoint_path = join(dir_to_download, checkpoint)

            logger.info("Found predefined checkpoint: {}".format(checkpoint))
            # No need to do anything, file has been already downloaded.
            if not exists(target_checkpoint_path):
                logger.info("Downloading checkpoint to: {}".format(target_checkpoint_path))
                download(dest_file_path=target_checkpoint_path,
                         source_url=data["checkpoint"],
                         logger=logger)

            return data["state"], target_checkpoint_path, data["label_scaler"]

        return None, None, None

    @staticmethod
    def init_bert_model(pretrain_path, labels_scaler, ckpt_path, device_type, predefined, logger=None,
                        dir_to_donwload=None, pooler='cls', max_length=128, mask_entity=True):
        """ This is a main and core method for inference based on OpenNRE framework.
        """
        # Check predefined checkpoints for local downloading.
        predefined_pretrain_path, predefined_ckpt_path, ckpt_label_scaler = \
            BertOpenNREInferencePipelineItem.try_download_predefined_checkpoint(
                checkpoint=ckpt_path, dir_to_download=dir_to_donwload, predefined=predefined, logger=logger)

        # Update checkpoint and pretrain paths with the predefined.
        ckpt_path = predefined_ckpt_path if predefined_ckpt_path is not None else ckpt_path
        pretrain_path = predefined_pretrain_path if predefined_pretrain_path is not None else pretrain_path
        labels_scaler = ckpt_label_scaler if ckpt_label_scaler is not None else labels_scaler

        # Load original model.
        bert_encoder = BertOpenNREInferencePipelineItem.load_bert_sentence_encoder(
            pooler=pooler, mask_entity=mask_entity, max_length=max_length, pretrain_path=pretrain_path)
        # Load checkpoint.
        rel2id = BertOpenNREInferencePipelineItem.scaler_to_rel2id(labels_scaler)
        model = SoftmaxNN(bert_encoder, len(rel2id), rel2id)
        model.load_state_dict(torch.load(ckpt_path, map_location=torch.device(device_type))['state_dict'])
        return model

    @staticmethod
    def iter_results(parallel_model, eval_loader, data_ids):

        # It is important we should open database.
        with eval_loader.dataset:

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
                        yield data_ids[l_ind], int(pred[i].item())
                        l_ind += 1

    def __iter_predict_result(self, samples_filepath, batch_size):
        # Compose evaluator.
        sentence_eval = sentence_re_loader(path=samples_filepath,
                                           rel2id=self.__model.rel2id,
                                           tokenizer=self.__model.sentence_encoder.tokenize,
                                           batch_size=batch_size,
                                           table_name=self.__table_name,
                                           task_kwargs=self.__task_kwargs,
                                           num_workers=self.__data_loader_num_workers,
                                           shuffle=False)

        with sentence_eval.dataset as dataset:

            # Iter output results.
            results_it = self.iter_results(parallel_model=torch.nn.DataParallel(self.__model),
                                           data_ids=list(dataset.iter_ids()),
                                           eval_loader=sentence_eval)

            total = len(sentence_eval.dataset)

        return results_it, total

    def apply_core(self, input_data, pipeline_ctx):

        # Fetching the input data.
        labels_scaler = pipeline_ctx.provide("labels_scaler")

        # Try to obrain from the specific input variable.
        samples_filepath = pipeline_ctx.provide_or_none("opennre_samples_filepath")
        if samples_filepath is None:
            samples_io = pipeline_ctx.provide("samples_io")
            samples_filepath = samples_io.create_target(data_type=DataType.Test)

        # Initialize model if the latter has not been yet.
        if self.__model is None:

            ckpt_dir = pipeline_ctx.provide_or_none("opennre_ckpt_cache_dir")

            self.__model = self.init_bert_model(
                pretrain_path=self.__pretrained_bert,
                ckpt_path=self.__checkpoint_path,
                device_type=self.__device_type,
                max_length=self.__max_seq_length,
                pooler=self.__pooler,
                labels_scaler=labels_scaler,
                mask_entity=True,
                predefined=self.__predefined_ckpts,
                logger=self.__logger,
                dir_to_donwload=get_default_download_dir() if ckpt_dir is None else ckpt_dir)

        iter_infer, total = self.__iter_predict_result(samples_filepath=samples_filepath, batch_size=self.__batch_size)
        pipeline_ctx.update("iter_infer", iter_infer)
        pipeline_ctx.update("iter_total", total)
