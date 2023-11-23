from enum import Enum

from arekit.common.entities.str_fmt import StringEntitiesFormatter
from arekit.common.labels.scaler.base import BaseLabelScaler
from arekit.contrib.bert.input.providers.cropped_sample import CroppedBertSampleRowProvider
from arekit.contrib.bert.terms.mapper import BertDefaultStringTextTermsMapper

from arelight.samplers.types import BertSampleProviderTypes


class BertTextBRussianPrompts(Enum):
    """
    Default, based on COLA, but includes an extra text_b.
        text_b: Pseudo-sentence w/o S.P (S.P -- sentiment polarity)
        text_b: Question w/o S.P (S.P -- sentiment polarity)

    Multilabel variant

    Notation were taken from paper:
    https://www.aclweb.org/anthology/N19-1035.pdf
    """

    NLI = '{subject} к {object} в контексте : << {context} >>'

    QA = 'Что вы думаете по поводу отношения {subject} к {object} в контексте : << {context} >> ?'


def create_bert_sample_provider(provider_type, label_scaler, entity_formatter, crop_window):
    """ This is a factory method, which allows to instantiate any of the
        supported bert_sample_encoders
    """
    assert(isinstance(provider_type, BertSampleProviderTypes) or provider_type is None)
    assert(isinstance(label_scaler, BaseLabelScaler))
    assert(isinstance(entity_formatter, StringEntitiesFormatter))

    text_terms_mapper = BertDefaultStringTextTermsMapper(entity_formatter)

    text_b_prompt = None
    if provider_type == BertSampleProviderTypes.NLI_M:
        text_b_prompt = BertTextBRussianPrompts.NLI.value
    if provider_type == BertSampleProviderTypes.QA_M:
        text_b_prompt = BertTextBRussianPrompts.QA.value

    return CroppedBertSampleRowProvider(crop_window_size=crop_window,
                                        text_b_template=text_b_prompt,
                                        text_terms_mapper=text_terms_mapper,
                                        label_scaler=label_scaler)
