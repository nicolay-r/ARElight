import os

import utils
import torch
import unittest

from os.path import join

from opennre.encoder import BERTEncoder
from opennre.framework import SentenceRELoader
from opennre.model import SoftmaxNN

from arelight.pipelines.items.inference_bert_opennre import BertOpenNREInferencePipelineItem
from arelight.predict_provider import BasePredictProvider
from arelight.predict_writer_csv import TsvPredictWriter
from arelight.run.utils import OPENNRE_CHECKPOINTS


class TestLoadModel(unittest.TestCase):

    CKPT = "ra4-rsr1_DeepPavlov-rubert-base-cased_cls.pth.tar"

    def test_launch_model(self):
        pretrain_path, ckpt_path, label_scaler = BertOpenNREInferencePipelineItem.try_download_predefined_checkpoint(
            checkpoint=self.CKPT, predefined=OPENNRE_CHECKPOINTS, dir_to_download=utils.TEST_OUT_DIR)
        model = BERTEncoder(pretrain_path=pretrain_path, mask_entity=True, max_length=512)
        rel2id = BertOpenNREInferencePipelineItem.scaler_to_rel2id(label_scaler)
        model = SoftmaxNN(model, len(rel2id), rel2id)
        model.load_state_dict(torch.load(ckpt_path, map_location=torch.device('cpu'))["state_dict"])

    def test_infer(self):
        self.infer_bert(pretrain_path=None,
                        ckpt_path=self.CKPT,
                        labels_scaler=None,
                        predefined=OPENNRE_CHECKPOINTS,
                        output_file_gzip=join(utils.TEST_OUT_DIR, "opennre-data-test.tsv.gz"))

    @staticmethod
    def infer_bert(pretrain_path, labels_scaler, output_file_gzip, predefined, ckpt_path=None, pooler='cls',
                   batch_size=6, max_length=128, mask_entity=True):

        test_data_file = join(utils.TEST_DATA_DIR, "opennre-data-test-predict.json")

        model = BertOpenNREInferencePipelineItem.init_bert_model(
            pretrain_path=pretrain_path, labels_scaler=labels_scaler, ckpt_path=ckpt_path,
            device_type="cpu", max_length=max_length, mask_entity=mask_entity,
            dir_to_donwload=os.getcwd(), pooler=pooler, predefined=predefined)

        eval_loader = SentenceRELoader(test_data_file,
                                       model.rel2id,
                                       model.sentence_encoder.tokenize,
                                       batch_size,
                                       False)

        it_results = BertOpenNREInferencePipelineItem.iter_results(
            parallel_model=torch.nn.DataParallel(model),
            data_ids=list(BertOpenNREInferencePipelineItem.extract_ids(test_data_file)),
            eval_loader=eval_loader)

        # Gathering the content
        title, contents_it = BasePredictProvider().provide(
            sample_id_with_uint_labels_iter=it_results,
            labels_count=3)

        w = TsvPredictWriter()
        w.set_target(output_file_gzip)
        with w:
            w.write(title=title, contents_it=contents_it)
