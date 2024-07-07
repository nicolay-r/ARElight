from arekit.contrib.utils.data.readers.base import BaseReader


class CustomSQliteReader(BaseReader):

    def __init__(self, storage_type, **storage_kwargs):
        self._storage_kwargs = storage_kwargs
        self._storage_type = storage_type

    def extension(self):
        return ".sqlite"

    def read(self, target):
        return self._storage_type(path=target, **self._storage_kwargs)
