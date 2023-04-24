from arekit.common.bound import Bound
from arekit.common.entities.base import Entity
from arekit.common.news.objects_parser import SentenceObjectsParserPipelineItem
from arekit.common.text.partitioning.terms import TermsPartitioning

from arelight.ner.deep_pavlov import DeepPavlovNER
from arelight.ner.obj_desc import NerObjectDescriptor


class DeepPavlovNERPipelineItem(SentenceObjectsParserPipelineItem):

    def __init__(self, obj_filter=None, ner_model_cfg=None):
        assert(callable(obj_filter) or obj_filter is None)

        # Initialize bert-based model instance.
        self.__dp_ner = DeepPavlovNER(ner_model_cfg)
        self.__obj_filter = obj_filter
        super(DeepPavlovNERPipelineItem, self).__init__(TermsPartitioning())

    def _get_parts_provider_func(self, input_data, pipeline_ctx):
        return self.__iter_subs_values_with_bounds(input_data)

    def __iter_subs_values_with_bounds(self, terms_list):
        assert(isinstance(terms_list, list))

        single_sequence = [terms_list]
        processed_sequences = self.__dp_ner.extract(sequences=single_sequence)

        for p_sequence in processed_sequences:
            for s_obj in p_sequence:
                assert(isinstance(s_obj, NerObjectDescriptor))

                if self.__obj_filter is not None and not self.__obj_filter(s_obj):
                    continue

                value = " ".join(terms_list[s_obj.Position:s_obj.Position + s_obj.Length])
                entity = Entity(value=value, e_type=s_obj.ObjectType)
                yield entity, Bound(pos=s_obj.Position, length=s_obj.Length)
