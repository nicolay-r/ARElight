import json
import os

import utils
import unittest
from os.path import join

import torch
from opennre.encoder import BERTEncoder
from opennre.framework import SentenceRELoader
from opennre.model import SoftmaxNN

from arelight.pipelines.items.inference_bert_opennre import BertOpenNREInferencePipelineItem
from arelight.pipelines.items.utils import try_download_predefined_checkpoint
from arelight.predict_provider import BasePredictProvider
from arelight.predict_writer_csv import TsvPredictWriter


class TestLoadModel(unittest.TestCase):

    def test_launch_model(self):
        rel2id = json.loads('{"0":0,"1":1,"2":2}')
        pretrain_path, ckpt_path = try_download_predefined_checkpoint(
            checkpoint="ra4-rsr1_DeepPavlov-rubert-base-cased_cls.pth.tar",
            dir_to_download=utils.TEST_OUT_DIR)
        model = BERTEncoder(pretrain_path=pretrain_path, mask_entity=True, max_length=512)
        model = SoftmaxNN(model, len(rel2id), rel2id)
        model.load_state_dict(torch.load(ckpt_path, map_location=torch.device('cpu'))["state_dict"])

    def test_infer(self):
        pretrain_path, ckpt_path = try_download_predefined_checkpoint(
            checkpoint="ra4-rsr1_DeepPavlov-rubert-base-cased_cls.pth.tar",
            dir_to_download=utils.TEST_OUT_DIR)

        self.infer_bert(pretrain_path=pretrain_path,
                        ckpt_source=ckpt_path,
                        rel2id=json.loads('{"0":0,"1":1,"2":2}'),
                        output_file_gzip=join(utils.TEST_OUT_DIR, "opennre-data-test.tsv.gz"))

    @staticmethod
    def infer_bert(pretrain_path, rel2id, output_file_gzip, ckpt_source=None, pooler='cls',
                   batch_size=6, max_length=128, mask_entity=True):

        test_data_file = join(utils.TEST_DATA_DIR, "opennre-data-test-predict.json")

        model = BertOpenNREInferencePipelineItem.init_bert_model(
            pretrain_path=pretrain_path, rel2id=rel2id, ckpt_path=ckpt_source,
            device_type="cpu", max_length=max_length, mask_entity=mask_entity,
            dir_to_donwload=os.getcwd(), pooler=pooler)

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
