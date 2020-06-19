# Python hack to allow definition of "infix" operators using the syntax <<op>> or |op|

class Infix:
    def __init__(self, function):
        self.function = function
    def __ror__(self, other):
        return Infix(lambda x, self=self, other=other: self.function(other, x))
    def __or__(self, other):
        return self.function(other)
    def __rlshift__(self, other):
        return Infix(lambda x, self=self, other=other: self.function(other, x))
    def __rshift__(self, other):
        return self.function(other)
    def __call__(self, value1, value2):
        return self.function(value1, value2)

coalesce = Infix(lambda val, default: val if not val is None else default)

class Nullable:
    def __init__(self):
        pass
    def __get_generic(self):


class Nullable_Type(Nullable):
    def __init__(self, prop):
        super().__init()
