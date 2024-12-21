from arekit.common.bound import Bound
from arekit.common.pipeline.items.base import BasePipelineItem
from arekit.common.text.partitioning import Partitioning
from arekit.common.utils import split_by_whitespaces

from arelight.pipelines.items.entity import IndexedEntity
from arelight.utils import IdAssigner, auto_import


class TransformersNERPipelineItem(BasePipelineItem):

    def __init__(self, id_assigner, ner_model_name, device, obj_filter=None, display_value_func=None, **kwargs):
        """ chunk_limit: int
                length of text part in words that is going to be provided in input.
        """
        assert(callable(obj_filter) or obj_filter is None)
        assert(isinstance(id_assigner, IdAssigner))
        assert(callable(display_value_func) or display_value_func is None)
        super(TransformersNERPipelineItem, self).__init__(**kwargs)

        # Setup third-party modules.
        model_init = auto_import("arelight.third_party.transformers.init_token_classification_model")
        self.annotate_ner = auto_import("arelight.third_party.transformers.annotate_ner")

        # Transformers-related parameters.
        self.__device = device
        self.__model, self.__tokenizer = model_init(model_path=ner_model_name, device=self.__device)

        # Initialize bert-based model instance.
        self.__obj_filter = obj_filter
        self.__id_assigner = id_assigner
        self.__disp_value_func = display_value_func
        self.__partitioning = Partitioning(text_fmt="str")

    # region Private methods

    def __get_parts_provider_func(self, input_data):
        assert(isinstance(input_data, str))
        parts = self.annotate_ner(model=self.__model, tokenizer=self.__tokenizer, text=input_data,
                                  device=self.__device)
        for entity, bound in self.__iter_parsed_entities(parts):
            yield entity, bound

    def __iter_parsed_entities(self, parts):
        for p in parts:
            assert (isinstance(p, dict))
            value = p["word"]

            if len(value) == 0:
                continue

            if self.__obj_filter is not None and not self.__obj_filter(p["entity_group"]):
                continue

            entity = IndexedEntity(
                value=value, e_type=p["entity_group"], entity_id=self.__id_assigner.get_id(),
                display_value=self.__disp_value_func(value) if self.__disp_value_func is not None else None)

            yield entity, Bound(pos=p["start"], length=p["end"] - p["start"])

    @staticmethod
    def __iter_fixed_terms(terms):
        for e in terms:
            if isinstance(e, str):
                for term in split_by_whitespaces(e):
                    yield term
            else:
                yield e

    # endregion

    def apply_core(self, input_data, pipeline_ctx):
        parts_it = self.__get_parts_provider_func(input_data)
        handled = self.__partitioning.provide(text=input_data, parts_it=parts_it)
        return list(self.__iter_fixed_terms(handled))
