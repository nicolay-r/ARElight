from arekit.common.pipeline.items.base import BasePipelineItem
from arelight.pipelines.items.entity import IndexedEntity
from arelight.utils import IdAssigner


class TextEntitiesParser(BasePipelineItem):

    def __init__(self, id_assigner, display_value_func=None, **kwargs):
        assert(isinstance(id_assigner, IdAssigner))
        assert(callable(display_value_func) or display_value_func is None)
        super(TextEntitiesParser, self).__init__(**kwargs)
        self.__id_assigner = id_assigner
        self.__disp_value_func = display_value_func

    def __process_word(self, word):
        assert(isinstance(word, str))

        # If this is a special word which is related to the [entity] mention.
        if word[0] == "[" and word[-1] == "]":
            value = word[1:-1]
            entity = IndexedEntity(value=value, e_type="UNDEFINED", entity_id=self.__id_assigner.get_id(),
                                   display_value=self.__disp_value_func(value) if self.__disp_value_func is not None else None)
            return entity

        return word

    def apply_core(self, input_data, pipeline_ctx):
        assert(isinstance(input_data, list))
        return [self.__process_word(w) for w in input_data]
