import asyncio
import collections
import csv
import importlib

from bulk_chain.api import iter_content
from bulk_chain.core.utils import dynamic_init


def auto_import(name):
    """ Import from the external python packages.
    """
    def __get_module(comps_list):
        return importlib.import_module(".".join(comps_list))

    components = name.split('.')
    return getattr(__get_module(components[:-1]), components[-1])


def flatten(xss):
    l = []
    for xs in xss:
        if isinstance(xs, collections.abc.Iterable):
            for x in xs:
                l.append(x)
        else:
            l.append(xs)
    return l


def iter_csv_lines(csv_file, column_name, delimiter=","):

    with csv_file:
        csv_reader = csv.DictReader(csv_file, delimiter=delimiter)

        if column_name not in csv_reader.fieldnames:
            print(f"Error: {column_name} column not found.")

        for row in csv_reader:
            yield row[column_name]


def init_llm(class_filepath, api_key, model_name="meta/meta-llama-3-70b-instruct"):
    return dynamic_init(class_filepath)(
        api_token=api_key,
        model_name=model_name)


def infer_async_batch_it(**kwargs):
    c = iter_content(infer_mode="batch_async", **kwargs)
    for batch in c:
        for record in batch:
            yield record


def batch_iter(arr, block_size):
    if block_size <= 0:
        raise ValueError("block_size must be positive")

    for i in range(0, len(arr), block_size):
        yield arr[i:i + block_size]


def get_event_loop():
    try:
        event_loop = asyncio.get_event_loop()
    except RuntimeError as e:
        if str(e).startswith('There is no current event loop in thread'):
            event_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(event_loop)
        else:
            raise

    return event_loop
