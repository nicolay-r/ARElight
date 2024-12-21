from arekit.contrib.utils.data.storages.jsonl_based import JsonlBasedRowsStorage

from arelight.readers.base import BaseReader


class JsonlReader(BaseReader):

    def extension(self):
        return ".jsonl"

    def read(self, target):
        rows = []
        with open(target, "r") as f:
            for line in f.readlines():
                rows.append(line)
        return JsonlBasedRowsStorage(rows)
