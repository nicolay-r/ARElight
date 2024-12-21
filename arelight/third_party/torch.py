import logging

import torch
from torch.utils import data

from arelight.third_party.sqlite3 import SQLite3Service


class SQLiteSentenceREDataset(data.Dataset):
    """ Sentence-level relation extraction dataset
        This is a original OpenNRE implementation, adapted for SQLite.
    """

    def __init__(self, path, table_name, rel2id, tokenizer, kwargs, task_kwargs):
        """
        Args:
            path: path of the input file sqlite file
            rel2id: dictionary of relation->id mapping
            tokenizer: function of tokenizing
        """
        assert(isinstance(task_kwargs, dict))
        assert("no_label" in task_kwargs)
        assert("default_id_column" in task_kwargs)
        assert("index_columns" in task_kwargs)
        assert("text_columns" in task_kwargs)
        super().__init__()
        self.path = path
        self.tokenizer = tokenizer
        self.rel2id = rel2id
        self.kwargs = kwargs
        self.task_kwargs = task_kwargs
        self.table_name = table_name
        self.sqlite_service = SQLite3Service()

    def iter_ids(self, id_column=None):
        col_name = self.task_kwargs["default_id_column"] if id_column is None else id_column
        for row in self.sqlite_service.iter_rows(select_columns=col_name, table_name=self.table_name):
            yield row[0]

    def __len__(self):
        return self.sqlite_service.table_rows_count(self.table_name)

    def __getitem__(self, index):
        found_text_columns = self.sqlite_service.get_column_names(
            table_name=self.table_name, filter_name=lambda col_name: col_name in self.task_kwargs["text_columns"])

        iter_rows = self.sqlite_service.iter_rows(
            select_columns=",".join(self.task_kwargs["index_columns"] + found_text_columns),
            value=index,
            column_value=self.task_kwargs["default_id_column"],
            table_name=self.table_name)

        fetched_row = next(iter_rows)

        opennre_item = {
            "text": " ".join(fetched_row[-len(found_text_columns):]),
            "h": {"pos": [fetched_row[0], fetched_row[0] + 1]},
            "t": {"pos": [fetched_row[1], fetched_row[1] + 1]},
        }

        seq = list(self.tokenizer(opennre_item, **self.kwargs))

        return [self.rel2id[self.task_kwargs["no_label"]]] + seq  # label, seq1, seq2, ...

    def collate_fn(data):
        data = list(zip(*data))
        labels = data[0]
        seqs = data[1:]
        batch_labels = torch.tensor(labels).long()  # (B)
        batch_seqs = []
        for seq in seqs:
            batch_seqs.append(torch.cat(seq, 0))  # (B, L)
        return [batch_labels] + batch_seqs

    def eval(self, pred_result, use_name=False):
        raise NotImplementedError()

    def __enter__(self):
        self.sqlite_service.connect(self.path)
        logging.info("Loaded sentence RE dataset {} with {} lines and {} relations.".format(
            self.path, len(self), len(self.rel2id)))
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.sqlite_service.disconnect()


def sentence_re_loader(path, table_name, rel2id, tokenizer, batch_size, shuffle,
                       task_kwargs, num_workers, collate_fn=SQLiteSentenceREDataset.collate_fn, **kwargs):
    dataset = SQLiteSentenceREDataset(path=path, table_name=table_name, rel2id=rel2id,
                                      tokenizer=tokenizer, kwargs=kwargs, task_kwargs=task_kwargs)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batch_size,
                                  shuffle=shuffle,
                                  pin_memory=True,
                                  num_workers=num_workers,
                                  collate_fn=collate_fn)
    return data_loader
