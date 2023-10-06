from arekit.common.entities.base import Entity


class IndexedEntity(Entity):
    """ Same as the base Entity but supports indexing.
    """

    def __init__(self, value, e_type, entity_id, display_value=None):
        super(IndexedEntity, self).__init__(value=value, e_type=e_type, display_value=display_value)
        self.__id = entity_id

    @property
    def ID(self):
        return self.__id
