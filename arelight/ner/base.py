from arelight.ner.obj_desc import NerObjectDescriptor


class BaseNER(object):
    """ CoNLL format based Named Entity Extractor
        for list of input sequences, where sequence is a list of terms.
    """

    separator = '-'
    begin_tag = 'B'
    inner_tag = 'I'

    def extract(self, sequences):
        assert(isinstance(sequences, list))
        terms, labels = self._forward(sequences)
        return self.iter_descriptors(terms=terms, labels=labels)

    def iter_descriptors(self, terms, labels):
        assert(len(terms) == len(labels))
        for seq, tags in zip(terms, labels):
            objs_len = [len(entry) for entry in self.__iter_merged(seq, tags)]
            objs_type = [self.__tag_type(tag) for tag in tags if self.__tag_part(tag) == self.begin_tag]
            objs_positions = [j for j, tag in enumerate(tags) if self.__tag_part(tag) == self.begin_tag]

            descriptors = [NerObjectDescriptor(pos=objs_positions[i], length=objs_len[i], obj_type=objs_type[i])
                           for i in range(len(objs_len))]
            yield seq, descriptors

    def _forward(self, seqences):
        raise NotImplementedError()

    # region private methods

    def __iter_merged(self, terms, tags):
        buffer = None
        for i, tag in enumerate(tags):
            current_tag = self.__tag_part(tag)
            if current_tag == self.begin_tag:
                if buffer is not None:
                    yield buffer
                buffer = [terms[i]]
            elif current_tag == self.inner_tag and buffer is not None:
                buffer.append(terms[i])

        if buffer is not None:
            yield buffer

    @staticmethod
    def __tag_part(tag):
        assert(isinstance(tag, str))
        return tag if BaseNER.separator not in tag else tag[:tag.index(BaseNER.separator)]

    @staticmethod
    def __tag_type(tag):
        assert(isinstance(tag, str))
        return "" if BaseNER.separator not in tag else tag[tag.index(BaseNER.separator) + 1:]

    # endregion
