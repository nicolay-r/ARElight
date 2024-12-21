from arekit.contrib.utils.data.storages.sqlite_based import SQliteBasedRowsStorage

from arelight.readers.base import BaseReader


class SQliteReader(BaseReader):

    def __init__(self, table_name):
        self.__table_name = table_name

    def extension(self):
        return ".sqlite"

    def read(self, target):
        return SQliteBasedRowsStorage(path=target, table_name=self.__table_name)
