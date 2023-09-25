from arekit.common.data import const
from arekit.common.data.input.providers.text.single import BaseSingleTextProvider
from arekit.common.experiment.data_type import DataType
from arekit.common.labels.base import NoLabel
from arekit.contrib.bert.input.providers.text_pair import PairTextProvider
from arekit.contrib.utils.data.readers.csv_pd import PandasCsvReader
from arekit.contrib.utils.data.storages.row_cache import RowCacheStorage
from arekit.contrib.utils.data.writers.csv_native import NativeCsvWriter
from arekit.contrib.utils.data.writers.json_opennre import OpenNREJsonWriter
from arekit.contrib.utils.io_utils.samples import SamplesIO

from arelight.samplers.bert import create_bert_sample_provider
from arelight.samplers.types import SampleFormattersService


def get_samples_setup_settings(infer_engines,
                               output_dir,
                               labels_scaler=None,
                               entity_fmt=None,
                               text_b_type=SampleFormattersService.name_to_type("nli_m"),
                               samples_prefix="samples"):
    """ This is a default setup for sampling in demo.

        returns: dict
            dict of the parameters that are expected to be setted up.
    """

    if "opennre" in infer_engines:
        # OpenNRE supports the specific type of writer based on JSONL.
        writer = OpenNREJsonWriter(text_columns=[BaseSingleTextProvider.TEXT_A, PairTextProvider.TEXT_B],
                                   keep_extra_columns=False,
                                   # `0` basically.
                                   na_value=str(labels_scaler.label_to_uint(NoLabel())))
    else:
        writer = NativeCsvWriter(delimiter=',')

    # Setup SamplesIO.
    samples_io = SamplesIO(target_dir=output_dir,
                           reader=PandasCsvReader(sep=',', compression="infer"),
                           prefix=samples_prefix,
                           writer=writer)

    return {
        "rows_provider": create_bert_sample_provider(
            provider_type=text_b_type, label_scaler=labels_scaler, entity_formatter=entity_fmt),
        "samples_io": samples_io,
        "storage": RowCacheStorage(force_collect_columns=[
            const.ENTITIES, const.ENTITY_VALUES, const.ENTITY_TYPES, const.SENT_IND
        ]),
        "save_labels_func": lambda data_type: data_type != DataType.Test
    }