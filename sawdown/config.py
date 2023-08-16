import enum


class BinaryMappingMode(enum.IntEnum):
    ZERO_ONE = 1
    ONES = 2


class Config(object):

    def __init__(self, initialization_max_iters=500, initialization_decay_steps=20,
                 binary_mapping_mode=BinaryMappingMode.ZERO_ONE,
                 parallelization=0):
        self.initialization_max_iters = initialization_max_iters
        self.initialization_decay_steps = initialization_decay_steps
        self.binary_mapping_mode = binary_mapping_mode
        self.parallelization = parallelization

    def binary_bounds(self):
        if self.binary_mapping_mode == BinaryMappingMode.ZERO_ONE:
            return 0., 1.
        return -1., 1.
