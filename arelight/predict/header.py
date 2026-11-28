from arekit.common.data import const


class PredictHeader:

    @staticmethod
    def _uint_label_to_str(uint_label):
        return f"col_{uint_label}"

    @staticmethod
    def create_header(uint_labels, uint_label_to_str=None, create_id=True):
        if uint_label_to_str is None:
            uint_label_to_str = PredictHeader._uint_label_to_str
        assert(callable(uint_label_to_str))

        header = []

        if create_id:
            header.append(const.ID)

        header.extend([uint_label_to_str(uint_label) for uint_label in uint_labels])

        return header
