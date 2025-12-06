from arekit.common.data.input.providers.text.single import BaseSingleTextProvider
from arekit.common.data.input.terms_mapper import OpinionContainingTextTermsMapper
from arekit.common.labels.scaler.base import BaseLabelScaler
from arekit.contrib.prompt.sample import PromptedSampleRowProvider

from arelight.arekit.indexed_entity import IndexedEntity


def create_prompted_sample_provider(label_scaler, crop_window, **mapper_kwargs):
    assert(isinstance(label_scaler, BaseLabelScaler))
    return PromptedSampleRowProvider(
        prompt="{text}",
        crop_window_size=crop_window,
        is_entity_func=lambda term: isinstance(term, IndexedEntity),
        text_provider=BaseSingleTextProvider(text_terms_mapper=OpinionContainingTextTermsMapper(**mapper_kwargs)),
        label_scaler=label_scaler)
