import logging
import sqlite3

import torch
from torch.utils import data


class SQLiteSentenceREDataset(data.Dataset):
    """ Sentence-level relation extraction dataset
        This is a original OpenNRE implementation, adapted for SQLite.
    """

    def __init__(self, path, table_name, rel2id, tokenizer, kwargs, text_columns=None):
        """
        Args:
            path: path of the input file sqlite file
            rel2id: dictionary of relation->id mapping
            tokenizer: function of tokenizing
        """
        super().__init__()
        self.path = path
        self.tokenizer = tokenizer
        self.rel2id = rel2id
        self.kwargs = kwargs
        self.table_name = table_name
        # Task-related parameters.
        self.no_label = "0"
        self.default_id_column = "id"
        self.index_columns = ["s_ind", "t_ind"]
        self.text_columns = text_columns

    def iter_ids(self, id_column=None):
        col_name = self.default_id_column if id_column is None else id_column
        for row in self.conn.execute(f"select {col_name} from {self.table_name}").fetchall():
            yield row[0]

    def __len__(self):
        count_response = self.conn.execute(f"select count(*) from {self.table_name}").fetchone()
        return count_response[0]

    def __getitem__(self, index):

        # Automatically assign column names.
        # This is expected to be refactored as a task-specific approach of text columns assignation.
        if self.text_columns is None:
            cursor = self.conn.execute(f'select * from {self.table_name}')
            column_names = list(map(lambda x: x[0], cursor.description))
            self.text_columns = [col_name for col_name in column_names if col_name.startswith("text")]

        cols = ",".join(self.index_columns + self.text_columns)

        fetched_row = self.conn.execute(
            f"select {cols} from {self.table_name} where ({self.default_id_column} = ?)", (index,)).fetchone()

        opennre_item = {
            "text": " ".join(fetched_row[-len(self.text_columns):]),
            "h": {"pos": [fetched_row[0], fetched_row[0] + 1]},
            "t": {"pos": [fetched_row[1], fetched_row[1] + 1]},
        }

        seq = list(self.tokenizer(opennre_item, **self.kwargs))

        return [self.rel2id[self.no_label]] + seq  # label, seq1, seq2, ...

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
        self.conn = sqlite3.connect(self.path)
        logging.info("Loaded sentence RE dataset {} with {} lines and {} relations.".format(
            self.path, len(self), len(self.rel2id)))
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.conn.close()


def sentence_re_loader(path, table_name, rel2id, tokenizer, batch_size,
                       shuffle, text_columns=None, num_workers=8, collate_fn=SQLiteSentenceREDataset.collate_fn,
                       **kwargs):
    dataset = SQLiteSentenceREDataset(path=path, table_name=table_name, rel2id=rel2id,
                                      tokenizer=tokenizer, kwargs=kwargs, text_columns=text_columns)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batch_size,
                                  shuffle=shuffle,
                                  pin_memory=True,
                                  num_workers=num_workers,
                                  collate_fn=collate_fn)
    return data_loader
