import logging

from arekit.common.service.sqlite import SQLiteProvider
from arekit.common.utils import progress_bar_defined

from arelight.predict.writer import BasePredictWriter

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class SQLite3PredictWriter(BasePredictWriter):

    def __init__(self, table_name):
        super(SQLite3PredictWriter, self).__init__()
        self.__table_name = table_name

    def write(self, header, contents_it, total=None):

        content_header = header[1:]
        SQLiteProvider.write(
            columns=[f"col_{col_name}" for col_name in content_header],
            target=self._target,
            table_name=self.__table_name,
            data2col_func=lambda data: data,
            data_it=progress_bar_defined(iterable=map(lambda item: [item[0], item[1:]], contents_it),
                                         desc=f'Writing output (sqlite:{self.__table_name})',
                                         unit='rows', total=total),
            sqlite3_column_types=["INTEGER"] * len(content_header),
            id_column_name=header[0],
            id_column_type="INTEGER")

    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass
