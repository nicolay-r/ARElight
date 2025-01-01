import logging
import utils
import torch
import unittest

from tqdm import tqdm

from opennre.encoder import BERTEncoder
from opennre.model import SoftmaxNN

from arelight.pipelines.items.inference_bert_opennre import BertOpenNREInferencePipelineItem
from arelight.run.utils import OPENNRE_CHECKPOINTS
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
        self.infer_no_batch(pretrain_path=None,
                            ckpt_path=self.CKPT,
                            labels_scaler=None,
                            predefined=OPENNRE_CHECKPOINTS,
                            logger=logging.getLogger(__name__))

    @staticmethod
    def infer_no_batch(pretrain_path, labels_scaler, predefined, logger, ckpt_path=None,
                       pooler='cls', max_length=128, mask_entity=True):
        model = BertOpenNREInferencePipelineItem.init_bert_model(
            pretrain_path=pretrain_path, labels_scaler=labels_scaler, ckpt_path=ckpt_path,
            device_type="cpu", max_length=max_length, mask_entity=mask_entity, logger=logger,
            dir_to_donwload=get_default_download_dir(), pooler=pooler, predefined=predefined)

        texts = [
            {"token": "сша ввела санкции против россии".split(), "h": {"pos": [0, 1]}, "t": {"pos": [3, 4]}},
            {"token": "сша поддержала россии".split(), "h": {"pos": [0, 1]}, "t": {"pos": [3, 4]}},
            {"token": "сша и россия".split(), "h": {"pos": [0, 1]}, "t": {"pos": [3, 4]}},
        ]

        texts = texts * 100

        for input in tqdm(texts):
            _ = model.infer(input)
