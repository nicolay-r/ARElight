from arekit.common.pipeline.items.base import BasePipelineItem
from arelight.pipelines.items.entity import IndexedEntity


class TextEntitiesParser(BasePipelineItem):

    def __init__(self):
        super(TextEntitiesParser, self).__init__()
        self.__entities_registered = 0

    def __process_word(self, word):
        assert(isinstance(word, str))

        # If this is a special word which is related to the [entity] mention.
        if word[0] == "[" and word[-1] == "]":
            entity = IndexedEntity(value=word[1:-1], e_type="UNDEFINED", entity_id=self.__entities_registered)
            self.__entities_registered += 1
            return entity

        return word

    def apply_core(self, input_data, pipeline_ctx):
        assert(isinstance(input_data, list))
        self.__entities_registered = 0
        return [self.__process_word(w) for w in input_data]
