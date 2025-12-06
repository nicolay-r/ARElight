from arekit.common.entities.str_fmt import StringEntitiesFormatter
from arekit.common.entities.types import OpinionEntityType


class HighligtedEntitiesFormatter(StringEntitiesFormatter):

    def to_string(self, original_value, entity_type):
        assert(isinstance(entity_type, OpinionEntityType))

        if (entity_type == OpinionEntityType.Object) or (entity_type == OpinionEntityType.SynonymObject):
            return f"<<{original_value.Value}>> [OBJECT]"
        elif (entity_type == OpinionEntityType.Subject) or (entity_type == OpinionEntityType.SynonymSubject):
            return f"<<{original_value.Value}>> [SUBJECT]"
        elif entity_type == OpinionEntityType.Other:
            return f"<<{original_value.Value}>>"
