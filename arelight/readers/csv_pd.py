import importlib

from arelight.arekit.storages.pandas_based import PandasBasedRowsStorage
from arelight.readers.base import BaseReader


class PandasCsvReader(BaseReader):
    """ Represents a CSV-based reader, implmented via pandas API.
    """

    def __init__(self, sep='\t', header='infer', compression='infer', encoding='utf-8', col_types=None,
                 custom_extension=None):
        self.__sep = sep
        self.__compression = compression
        self.__encoding = encoding
        self.__header = header
        self.__custom_extension = custom_extension

        # Special assignation of types for certain columns.
        self.__col_types = col_types
        if self.__col_types is None:
            self.__col_types = dict()

    def extension(self):
        return ".tsv.gz" if self.__custom_extension is None else self.__custom_extension

    def __from_csv(self, filepath):
        pd = importlib.import_module("pandas")
        return pd.read_csv(filepath,
                           sep=self.__sep,
                           encoding=self.__encoding,
                           compression=self.__compression,
                           dtype=self.__col_types,
                           header=self.__header)

    def read(self, target):
        df = self.__from_csv(filepath=target)
        return PandasBasedRowsStorage(df)