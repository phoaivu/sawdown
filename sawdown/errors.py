class InitializationError(RuntimeError):
    def __init__(self, reason=''):
        self.reason = reason


class IncompatibleConstraintsError(RuntimeError):

    def __init__(self, first_type, second_type):
        self.first_type = first_type
        self.second_type = second_type
