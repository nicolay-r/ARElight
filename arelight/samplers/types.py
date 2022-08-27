from enum import Enum

from examples.utils import EnumConversionService


class BertSampleProviderTypes(Enum):
    """
    Supported format types.
    """

    """
    Default formatter
    """
    CLASSIF_M = 0

    """
    Natural Language Inference samplers
    paper: https://www.aclweb.org/anthology/N19-1035.pdf
    """
    QA_M = 1
    NLI_M = 2


class SampleFormattersService(EnumConversionService):

    _data = {
        'c_m': BertSampleProviderTypes.CLASSIF_M,
        "qa_m": BertSampleProviderTypes.QA_M,
        'nli_m': BertSampleProviderTypes.NLI_M,
    }