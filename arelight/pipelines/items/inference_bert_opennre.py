import json
import os
from os.path import join, dirname

import torch
from arekit.common.experiment.data_type import DataType
from arekit.common.labels.scaler.base import BaseLabelScaler
from arekit.common.pipeline.context import PipelineContext
from arekit.common.pipeline.items.base import BasePipelineItem

from opennre.encoder import BERTEntityEncoder, BERTEncoder
from opennre.framework import SentenceRELoader
from opennre.model import SoftmaxNN

from arelight.pipelines.items.utils import try_download_predefined_checkpoints
from arelight.predict_provider import BasePredictProvider
from arelight.predict_writer_csv import TsvPredictWriter


class BertOpenNREInferencePipelineItem(BasePipelineItem):

    def __init__(self, pretrained_bert, checkpoint_path, labels_scaler, max_seq_length,
                 batch_size=10, device_type='cpu', dir_to_download=None):
        assert(isinstance(max_seq_length, int))
        assert(isinstance(labels_scaler, BaseLabelScaler))

        self.__predict_provider = BasePredictProvider()
        self.__batch_size = batch_size
        self.__device_type = device_type
        self.__labels_scaler = labels_scaler

        # We compose specific mapping required by opennre to perform labels mapping.
        rel2id = {}
        for l in labels_scaler.ordered_suppoted_labels():
            uint_label = labels_scaler.label_to_uint(l)
            rel2id[str(uint_label)] = uint_label

        # Load model
        self.__model = self.init_bert_model(pretrain_path=pretrained_bert,
                                            rel2id=rel2id,
                                            ckpt_source=checkpoint_path,
                                            pooler='cls',
                                            max_length=max_seq_length,
                                            mask_entity=True,
                                            device_type=device_type,
                                            dir_to_donwload=os.getcwd() if dir_to_download is None else dir_to_download)

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
    def init_bert_model(pretrain_path, rel2id, ckpt_source, device_type, dir_to_donwload=None,
                        pooler='cls', max_length=128, mask_entity=True):
        """ This is a main and core method for inference based on OpenNRE framework.
        """
        # Check predefined checkpoints for local downloading.
        try_download_predefined_checkpoints(checkpoint=ckpt_source, dir_to_download=dir_to_donwload)

        # Load original model.
        bert_encoder = BertOpenNREInferencePipelineItem.load_bert_sentence_encoder(
            pooler=pooler, mask_entity=mask_entity, max_length=max_length, pretrain_path=pretrain_path)
        # Load checkpoint.
        model = SoftmaxNN(bert_encoder, len(rel2id), rel2id)
        model.load_state_dict(torch.load(ckpt_source, map_location=torch.device(device_type))['state_dict'])
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

    def apply_core(self, input_data, pipeline_ctx):
        assert (isinstance(pipeline_ctx, PipelineContext))

        def __iter_predict_result():
            # Compose evaluator.
            sentence_eval = SentenceRELoader(path=samples_filepath,
                                             rel2id=self.__model.rel2id,
                                             tokenizer=self.__model.sentence_encoder.tokenize,
                                             batch_size=6,
                                             shuffle=False)

            # Iter output results.
            return self.iter_results(parallel_model=torch.nn.DataParallel(self.__model),
                                     data_ids=list(self.extract_ids(samples_filepath)),
                                     eval_loader=sentence_eval)

        # Fetch other required in further information from input_data.
        samples_io = input_data.provide("samples_io")
        samples_filepath = samples_io.create_target(data_type=DataType.Test)

        # Setup predicted result writer.
        tgt = pipeline_ctx.provide_or_none("predict_fp")
        if tgt is None:
            tgt = join(dirname(samples_filepath), "predict.tsv.gz")

        # Setup target filepath.
        writer = TsvPredictWriter()
        writer.set_target(tgt)

        # Update for further pipeline items.
        pipeline_ctx.update("predict_fp", tgt)

        # Gathering the content
        title, contents_it = self.__predict_provider.provide(
            sample_id_with_uint_labels_iter=__iter_predict_result(),
            labels_count=self.__labels_scaler.LabelsCount)

        with writer:
            writer.write(title=title, contents_it=contents_it)
