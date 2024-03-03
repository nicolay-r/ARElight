class ChunkIterator:

    def __init__(self, data_iter, batch_size, chunk_limit):
        assert(isinstance(batch_size, int) and batch_size > 0)
        self.__data_iter = data_iter
        self.__index = -1
        self.__batch_size = batch_size
        self.__chunk_limit = chunk_limit
        self.__buffer = []

    def __iter__(self):
        return self

    def __next__(self):
        while True:
            if len(self.__buffer) > 0:
                break
            try:
                data = next(self.__data_iter)
                self.__index += 1
            except StopIteration:
                break
            for chunk_start in range(0, len(data), self.__chunk_limit):
                chunk = data[chunk_start:chunk_start + self.__chunk_limit]
                self.__buffer.append([self.__index, chunk])

        if len(self.__buffer) > 0:
            return self.__buffer.pop(0)

        raise StopIteration

