import logging

import utils
import torch
import unittest

from os.path import join

from opennre.encoder import BERTEncoder
from opennre.model import SoftmaxNN

from arelight.pipelines.items.inference_bert_opennre import BertOpenNREInferencePipelineItem
from arelight.predict.provider import BasePredictProvider
from arelight.predict.writer_csv import TsvPredictWriter
from arelight.run.utils import OPENNRE_CHECKPOINTS
from arelight.third_party.torch import sentence_re_loader
from arelight.utils import get_default_download_dir

logger = logging.getLogger(__name__)


class TestLoadModel(unittest.TestCase):

    CKPT = "ra4-rsr1_DeepPavlov-rubert-base-cased_cls.pth.tar"

    def test_launch_model(self):
        pretrain_path, ckpt_path, label_scaler = BertOpenNREInferencePipelineItem.try_download_predefined_checkpoint(
            checkpoint=self.CKPT, predefined=OPENNRE_CHECKPOINTS, dir_to_download=utils.TEST_OUT_DIR, logger=logger)
        model = BERTEncoder(pretrain_path=pretrain_path, mask_entity=True, max_length=512)
        rel2id = BertOpenNREInferencePipelineItem.scaler_to_rel2id(label_scaler)
        model = SoftmaxNN(model, len(rel2id), rel2id)
        model.load_state_dict(torch.load(ckpt_path, map_location=torch.device('cpu'))["state_dict"])

    def test_infer(self):
        self.infer_bert(pretrain_path=None,
                        ckpt_path=self.CKPT,
                        labels_scaler=None,
                        predefined=OPENNRE_CHECKPOINTS,
                        output_file_gzip=join(utils.TEST_OUT_DIR, "opennre-data-test.tsv.gz"),
                        logger=logging.getLogger(__name__))

    @staticmethod
    def infer_bert(pretrain_path, labels_scaler, output_file_gzip, predefined, logger, ckpt_path=None,
                   pooler='cls', batch_size=6, max_length=128, mask_entity=True):

        test_data_file = join(utils.TEST_DATA_DIR, "opennre-data-test-predict.sqlite")

        model = BertOpenNREInferencePipelineItem.init_bert_model(
            pretrain_path=pretrain_path, labels_scaler=labels_scaler, ckpt_path=ckpt_path,
            device_type="cpu", max_length=max_length, mask_entity=mask_entity, logger=logger,
            dir_to_donwload=get_default_download_dir(), pooler=pooler, predefined=predefined)

        eval_loader = sentence_re_loader(path=test_data_file,
                                         table_name="contents",
                                         rel2id=model.rel2id,
                                         tokenizer=model.sentence_encoder.tokenize,
                                         batch_size=batch_size,
                                         task_kwargs={
                                            "no_label": "0",
                                            "default_id_column": "id",
                                            "index_columns": ["s_ind", "t_ind"],
                                            "text_columns": ["text_a", "text_b"]
                                         },
                                         shuffle=False,
                                         num_workers=0)

        # Open database.
        with eval_loader.dataset as dataset:

            it_results = BertOpenNREInferencePipelineItem.iter_results(
                parallel_model=torch.nn.DataParallel(model),
                data_ids=list(dataset.iter_ids()),
                eval_loader=eval_loader)

            # Gathering the content
            header, contents_it = BasePredictProvider.provide_to_storage(
                sample_id_with_uint_labels_iter=it_results,
                uint_labels=list(range(3)))

            w = TsvPredictWriter()
            w.set_target(output_file_gzip)
            with w:
                w.write(header=header, contents_it=contents_it)
