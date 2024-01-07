from arekit.common.data.storages.base import BaseRowsStorage


def iter_predicted_labels(predict_data, label_to_str, keep_ind=True):
    assert(isinstance(predict_data, BaseRowsStorage))
    assert(isinstance(label_to_str, dict))

    for res_ind, row in predict_data:

        rel_type = None
        for col_label, rel_name in label_to_str.items():
            print(row)
            if row[col_label] > 0:
                rel_type = rel_name
                break

        if keep_ind:
            yield res_ind, rel_type
        else:
            yield rel_type