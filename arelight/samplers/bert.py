from arekit.common.data.input.providers.label.multiple import MultipleLabelProvider
from arekit.common.data.input.providers.rows.samples import BaseSampleRowProvider
from arekit.common.data.input.providers.text.single import BaseSingleTextProvider
from arekit.common.entities.str_fmt import StringEntitiesFormatter
from arekit.common.labels.scaler.base import BaseLabelScaler
from arekit.common.labels.str_fmt import StringLabelsFormatter
from arekit.contrib.bert.input.providers.text_pair import PairTextProvider
from arekit.contrib.bert.terms.mapper import BertDefaultStringTextTermsMapper
from arekit.contrib.utils.bert.text_b_rus import BertTextBTemplates

from arelight.samplers.types import BertSampleProviderTypes


def create_bert_sample_provider(provider_type, label_scaler, text_b_labels_fmt, entity_formatter):
    """ This is a factory method, which allows to instantiate any of the
        supported bert_sample_encoders
    """
    assert(isinstance(provider_type, BertSampleProviderTypes))
    assert(isinstance(entity_formatter, StringEntitiesFormatter))
    assert(isinstance(text_b_labels_fmt, StringLabelsFormatter))
    assert(isinstance(label_scaler, BaseLabelScaler))

    text_terms_mapper = BertDefaultStringTextTermsMapper(entity_formatter)

    text_b_template = None
    if provider_type == BertSampleProviderTypes.NLI_M:
        text_b_template = BertTextBTemplates.NLI.value
    if provider_type == BertSampleProviderTypes.QA_M:
        text_b_template = BertTextBTemplates.QA.value

    text_provider = BaseSingleTextProvider(text_terms_mapper=text_terms_mapper) \
        if text_b_labels_fmt is None else PairTextProvider(text_b_template=text_b_template,
                                                           text_b_labels_fmt=text_b_labels_fmt,
                                                           text_terms_mapper=text_terms_mapper)

    return BaseSampleRowProvider(label_provider=MultipleLabelProvider(label_scaler),
                                 text_provider=text_provider)
