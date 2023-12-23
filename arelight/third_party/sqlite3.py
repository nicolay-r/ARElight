import sqlite3


class SQLite3Service(object):

    def __init__(self):
        self.conn = None

    def connect(self, sqlite_path):
        self.conn = sqlite3.connect(sqlite_path)

    def disconnect(self):
        assert(self.conn is not None)
        self.conn.close()

    def table_rows_count(self, table_name):
        count_response = self.conn.execute(f"select count(*) from {table_name}").fetchone()
        return count_response[0]

    def get_column_names(self, table_name, filter_name=None):
        cursor = self.conn.execute(f"select * from {table_name}")
        column_names = list(map(lambda x: x[0], cursor.description))
        return [col_name for col_name in column_names
                if filter_name is None or (filter_name is not None and filter_name(col_name))]

    def iter_rows(self, table_name, select_columns="*", column_value=None, value=None, return_dict=False):
        """ Returns array of the values in the same order as the one provided in `select_columns` parameter.
        """
        if column_value is not None and value is not None:
            cursor = self.conn.execute(
                f"select {select_columns} from {table_name} where ({column_value} = ?)", (value,))
        else:
            cursor = self.conn.execute(f"select {select_columns} from {table_name}")

        if not return_dict:
            for row in cursor.fetchall():
                yield row

        # Return as dictionary
        column_names = list(map(lambda x: x[0], cursor.description))
        for row in cursor.fetchall():
            yield {col_name: row[i] for i, col_name in enumerate(column_names)}
