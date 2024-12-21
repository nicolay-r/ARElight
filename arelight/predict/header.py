from arekit.common.data import const


class PredictHeader:

    @staticmethod
    def create_header(uint_labels, uint_label_to_str, create_id=True):
        assert(callable(uint_label_to_str))

        header = []

        if create_id:
            header.append(const.ID)

        header.extend([uint_label_to_str(uint_label) for uint_label in uint_labels])

        return header
