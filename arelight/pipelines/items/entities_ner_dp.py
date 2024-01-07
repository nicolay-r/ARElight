from arekit.common.bound import Bound
from arekit.common.pipeline.items.base import BasePipelineItem
from arekit.common.text.partitioning import Partitioning

from arelight.ner.deep_pavlov import DeepPavlovNER
from arelight.ner.obj_desc import NerObjectDescriptor
from arelight.pipelines.items.entity import IndexedEntity
from arelight.utils import IdAssigner


class DeepPavlovNERPipelineItem(BasePipelineItem):

    def __init__(self, id_assigner, ner_model_name, obj_filter=None,
                 chunk_limit=128, display_value_func=None, **kwargs):
        """ chunk_limit: int
                length of text part in words that is going to be provided in input.
        """
        assert(callable(obj_filter) or obj_filter is None)
        assert(isinstance(chunk_limit, int) and chunk_limit > 0)
        assert(isinstance(id_assigner, IdAssigner))
        assert(callable(display_value_func) or display_value_func is None)
        super(DeepPavlovNERPipelineItem, self).__init__(**kwargs)

        # Initialize bert-based model instance.
        self.__dp_ner = DeepPavlovNER(ner_model_name)
        self.__obj_filter = obj_filter
        self.__chunk_limit = chunk_limit
        self.__id_assigner = id_assigner
        self.__disp_value_func = display_value_func
        self.__partitioning = Partitioning(text_fmt="list")

    def __iter_subs_values_with_bounds(self, terms_list):
        assert(isinstance(terms_list, list))

        for chunk_start in range(0, len(terms_list), self.__chunk_limit):
            single_sentence_chunk = [terms_list[chunk_start:chunk_start+self.__chunk_limit]]

            # NOTE: in some cases, for example URL links or other long input words,
            # the overall behavior might result in exceeding the assumed threshold.
            # In order to completely prevent it, we consider to wrap the call
            # of NER module into try-catch block.
            try:
                processed_sequences = self.__dp_ner.extract(sequences=single_sentence_chunk)
            except RuntimeError:
                processed_sequences = []

            entities_it = self.__iter_parsed_entities(processed_sequences,
                                                      chunk_terms_list=single_sentence_chunk[0],
                                                      chunk_offset=chunk_start)

            for entity, bound in entities_it:
                yield entity, bound

    def __iter_parsed_entities(self, processed_sequences, chunk_terms_list, chunk_offset):
        for p_sequence in processed_sequences:
            for s_obj in p_sequence:
                assert (isinstance(s_obj, NerObjectDescriptor))

                if self.__obj_filter is not None and not self.__obj_filter(s_obj):
                    continue

                value = " ".join(chunk_terms_list[s_obj.Position:s_obj.Position + s_obj.Length])
                entity = IndexedEntity(
                    value=value, e_type=s_obj.ObjectType, entity_id=self.__id_assigner.get_id(),
                    display_value=self.__disp_value_func(value) if self.__disp_value_func is not None else None)
                yield entity, Bound(pos=chunk_offset + s_obj.Position, length=s_obj.Length)

    def apply_core(self, input_data, pipeline_ctx):
        parts_it = self.__iter_subs_values_with_bounds(input_data)
        return self.__partitioning.provide(text=input_data, parts_it=parts_it)
