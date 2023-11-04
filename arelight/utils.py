import csv
import importlib
import os
import sys


def auto_import(name):
    """ Import from the external python packages.
    """
    def __get_module(comps_list):
        return importlib.import_module(".".join(comps_list))

    components = name.split('.')
    return getattr(__get_module(components[:-1]), components[-1])


def get_default_download_dir():
    """ Refered to NLTK toolkit approach
        https://github.com/nltk/nltk/blob/8e771679cee1b4a9540633cc3ea17f4421ffd6c0/nltk/downloader.py#L1051
    """

    # On Windows, use %APPDATA%
    if sys.platform == "win32" and "APPDATA" in os.environ:
        homedir = os.environ["APPDATA"]

    # Otherwise, install in the user's home directory.
    else:
        homedir = os.path.expanduser("~/")
        if homedir == "~/":
            raise ValueError("Could not find a default download directory")

    return os.path.join(homedir, ".arelight")


class IdAssigner(object):

    def __init__(self):
        self.__id = 0

    def get_id(self):
        curr_id = self.__id
        self.__id += 1
        return curr_id


def iter_csv_lines(csv_filepath, column_name, delimiter=","):

    with open(csv_filepath, mode='r', encoding='utf-8-sig') as csv_file:
        csv_reader = csv.DictReader(csv_file, delimiter=delimiter)

        if column_name not in csv_reader.fieldnames:
            print(f"Error: {column_name} column not found.")

        for row in csv_reader:
            yield row[column_name]
