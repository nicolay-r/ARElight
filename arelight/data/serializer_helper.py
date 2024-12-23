import logging

from collections.abc import Iterable

from arekit.common.data.input.providers.columns.sample import SampleColumnsProvider
from arekit.common.data.input.providers.rows.base import BaseRowProvider
from arekit.common.data.storages.base import BaseRowsStorage
from arekit.contrib.utils.data.contents.opinions import InputTextOpinionProvider

from arelight.data.repositories.base import BaseInputRepository
from arelight.data.repositories.sample import BaseInputSamplesRepository

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class InputDataSerializationHelper(object):

    @staticmethod
    def create_samples_repo(keep_labels, rows_provider, storage):
        assert(isinstance(rows_provider, BaseRowProvider))
        assert(isinstance(keep_labels, bool))
        assert(isinstance(storage, BaseRowsStorage))
        return BaseInputSamplesRepository(
            columns_provider=SampleColumnsProvider(store_labels=keep_labels),
            rows_provider=rows_provider,
            storage=storage)

    @staticmethod
    def fill_and_write(pipeline, repo, target, writer, doc_ids_iter, desc=""):
        assert(isinstance(pipeline, list))
        assert(isinstance(doc_ids_iter, Iterable))
        assert(isinstance(repo, BaseInputRepository))

        doc_ids = list(doc_ids_iter)

        repo.populate(contents_provider=InputTextOpinionProvider(pipeline),
                      doc_ids=doc_ids,
                      desc=desc,
                      writer=writer,
                      target=target)

        repo.push(writer=writer, target=target)
