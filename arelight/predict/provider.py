from collections.abc import Iterable

from arekit.common.data.storages.base import BaseRowsStorage

from arelight.predict.header import PredictHeader


class BasePredictProvider(object):

    UINT_TO_STR = lambda uint_label: f"col_{uint_label}"

    @staticmethod
    def __iter_contents(sample_id_with_uint_labels_iter, labels_count, column_extra_funcs):
        assert(isinstance(labels_count, int))

        for sample_id, uint_label in sample_id_with_uint_labels_iter:
            assert(isinstance(uint_label, int))

            labels = ['0'] * labels_count
            labels[uint_label] = '1'

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
    def provide_to_storage(sample_id_with_uint_labels_iter, uint_labels):
        assert(isinstance(sample_id_with_uint_labels_iter, Iterable))

        # Provide contents.
        contents_it = BasePredictProvider.__iter_contents(
            sample_id_with_uint_labels_iter=sample_id_with_uint_labels_iter,
            labels_count=len(uint_labels),
            column_extra_funcs=None)

        header = PredictHeader.create_header(uint_labels=uint_labels,
                                             uint_label_to_str=BasePredictProvider.UINT_TO_STR)

        return header, contents_it

    @staticmethod
    def iter_from_storage(predict_data, uint_labels, keep_ind=True):
        assert (isinstance(predict_data, BaseRowsStorage))

        header = PredictHeader.create_header(uint_labels=uint_labels,
                                             uint_label_to_str=BasePredictProvider.UINT_TO_STR,
                                             create_id=False)

        for res_ind, row in predict_data:

            uint_label = None
            for label, field_name in zip(uint_labels, header):
                if row[field_name] > 0:
                    uint_label = label
                    break

            if keep_ind:
                yield res_ind, uint_label
            else:
                yield uint_label
