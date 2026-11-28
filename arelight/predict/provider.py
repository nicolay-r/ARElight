from collections.abc import Iterable

from arekit.common.data.storages.base import BaseRowsStorage


class IntPredictProvider(object):

    @staticmethod
    def __iter_contents(data_it, uint_labels, column_extra_funcs):
        assert(isinstance(uint_labels, list))

        for data in data_it:
            sample_id, uint_label = data
            uint_label = int(str(uint_label).strip().strip("\"'"))
            assert(isinstance(uint_label, int))

            labels = ['0'] * len(uint_labels)
            labels[uint_labels.index(uint_label)] = '1'

            # Composing row contents.
            contents = [sample_id]

            # Optionally provide additional values.
            if column_extra_funcs is not None:
                for _, value_func in column_extra_funcs:
                    contents.append(str(value_func(sample_id)))

            # Providing row labels.
            contents.extend(labels)
            yield contents

    @staticmethod
    def handle_iter(data_it, uint_labels):
        assert(isinstance(data_it, Iterable))

        return IntPredictProvider.__iter_contents(
            data_it=data_it,
            uint_labels=list(uint_labels),
            column_extra_funcs=None)

    @staticmethod
    def iter_from_storage(predict_data, uint_labels, label_columns, keep_ind=True):
        assert (isinstance(predict_data, BaseRowsStorage))
        assert (isinstance(label_columns, list))

        for res_ind, row in predict_data:

            uint_label = None
            for label, field_name in zip(uint_labels, label_columns):
                if row[field_name] > 0:
                    uint_label = label
                    break

            if keep_ind:
                yield res_ind, uint_label
            else:
                yield uint_label
