import sqlite3

from arekit.common.data.const import ID
from arekit.common.data.storages.base import BaseRowsStorage


class JoinedSQliteBasedRowsStorage(BaseRowsStorage):

    def __init__(self, path, table_name_a, table_name_b, **kwargs):
        super(JoinedSQliteBasedRowsStorage, self).__init__(**kwargs)
        self.__path = path
        self.__table_name_a = table_name_a
        self.__table_name_b = table_name_b
        self.__conn = None

    def _iter_rows(self):
        with sqlite3.connect(self.__path) as conn:
            cursor = conn.execute(f"select * from {self.__table_name_a} inner join {self.__table_name_b}"
                                  f" on {self.__table_name_a}.{ID}={self.__table_name_b}.{ID}")
            for row_index, row in enumerate(cursor.fetchall()):
                row_dict = {cursor.description[i][0]: value for i, value in enumerate(row)}
                yield row_index, row_dict
