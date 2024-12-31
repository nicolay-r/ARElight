import logging
import sys
import unittest
from os.path import join, dirname, realpath

import torch

from arelight.pipelines.items.inference_bert_opennre import BertOpenNREInferencePipelineItem
from arelight.predict.provider import BasePredictProvider
from arelight.predict.writer_csv import TsvPredictWriter
from arelight.run.utils import OPENNRE_CHECKPOINTS
from arelight.third_party.legacy.torch import sentence_re_loader
from arelight.utils import get_default_download_dir

sys.path.append("..")

current_dir = dirname(realpath(__file__))
TEST_DATA_DIR = join(current_dir, "..", "data")
TEST_OUT_DIR = join("..", "_out")


class TestLoadModel(unittest.TestCase):

    CKPT = "ra4-rsr1_DeepPavlov-rubert-base-cased_cls.pth.tar"

    def test_infer(self):
        self.infer_bert(pretrain_path=None,
                        ckpt_path=self.CKPT,
                        labels_scaler=None,
                        predefined=OPENNRE_CHECKPOINTS,
                        output_file_gzip=join(TEST_OUT_DIR, "opennre-data-test.tsv.gz"),
                        logger=logging.getLogger(__name__))

    @staticmethod
    def infer_bert(pretrain_path, labels_scaler, output_file_gzip, predefined, logger, ckpt_path=None,
                   pooler='cls', batch_size=6, max_length=128, mask_entity=True):

        test_data_file = join(TEST_DATA_DIR, "opennre-data-test-predict.sqlite")

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
