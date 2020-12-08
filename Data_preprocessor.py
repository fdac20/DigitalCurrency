
#Parent class for a data generator. Do not use directly.Overwrite the _generator method to create a custom data generator.
class DataGenerator(object):
    def __init__(self, **gen_kwargs):
        self._trainable = False
        self.gen_kwargs = gen_kwargs
        DataGenerator.rewind(self)
        self.n_products = len(next(self)) / 2
        DataGenerator.rewind(self)

    def _generator(**kwargs):
        raise NotImplementedError()

    def __next__(self):
        try:
            return next(self.generator)
        except StopIteration as e:
            self._iterator_end()
            raise(e)

    def rewind(self):
        self.generator = self._generator(**self.gen_kwargs)

    def _iterator_end(self):
        pass
