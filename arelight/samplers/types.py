from enum import Enum

from arelight.run.utils import EnumConversionService


class BertSampleProviderTypes(Enum):
    """
    Natural Language Inference samplers
    paper: https://www.aclweb.org/anthology/N19-1035.pdf
    """
    QA_M = 1
    NLI_M = 2


class SampleFormattersService(EnumConversionService):

    _data = {
        "qa_m": BertSampleProviderTypes.QA_M,
        'nli_m': BertSampleProviderTypes.NLI_M,
    }