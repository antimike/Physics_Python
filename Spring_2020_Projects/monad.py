import infix
from abc import ABC, abstractmethod



# A restricted implementation of the monad idea.
# Any instance of a Monad-derived class is assumed to be writable as 'return(x)'
# for some x.
# Overloads >> to a monadic "bind."
# Can also model "impure" monad processes--e.g., in which some fn arguments are monadic
# but not others--via the << operator.
class Monad(ABC):
    def __init__(self, obj):
        self.__val = obj
        self.__compute__ = lambda x: x

    # This automatically satisfies the first monad law by virtue of the assumptions
    # that (1) any function we're binding to can be written as return * f for some "ordinary"
    # function f, and (2) that the state of the monad can be represented as return(x) for some
    # x at any time.
    def __pipeto__(self, fn):
        def inner(*args, **kwargs):
            self.__val = fn(self.__val, *args, **kwargs)
            return self
        self.__compute__ = inner
        return self
        # Delayed computation

    # Requires overriding implementation
    @abstractmethod
    @staticmethod
    def __bind__(val, fn):
        pass

    # Requires overriding implementation
    @abstractmethod
    @staticmethod
    def __return__(val):
        return val

    # Sugar for standard monadic "bind" operation via >> syntax
    def __rshift__(self, fn):
        self.__pipeto__(fn)

    # Can be used for partial monadic computations
    def __lshift__(self, dict):
        return self.__eval(**dict)


class Maybe(Monad):
    def __new__(cls, cls_name, bases, cls_dict):
        super().__new__(cls, cls_name, bases,cls_dict)



# Maybe type factory should take a value and return a "maybe"-type callable.
# Most importantly, it should implement "bind" correctly
# Should also enforce monad laws if possible

"""
In practice, the "result" of a monadic computation will be one or more of the following:
    - A series of side-effects (IO, data display, state manipulation, etc.)
    - A conversion back to "ordinary" types (e.g., Maybe --> {None, value_type} or List --> [])
"""

    """
    Requirements:
        - Bind (functions on type a to functions on type ma)
            - Overload >> operator?
        - Wrap (type constructor)

    Alternatives to metaclass implementation:
        - Higher-order fns on classes (i.e., decorators)
            - e.g., def maybe(cls): ... return wrapped_class
    Examples:
        - Continuation (i.e, callbacks)
        - Maybe
        - Plot / Display (?)
        - Extract / Import (?)
        - Validation (?)
    """
# book >> cont(extract) >> cont(display)

# Sig: Ma (a -> b) -> Mb
# Should fn be unary?
# Use interface to check for implementation of monad methods
def __bind_to_maybe(val, fn):
    def inner(*args, **kwargs):
        if val is None:
            return None
        else:
            try:
                _ = [x for x in val if x is None]

            return fn(val, args, kwargs)
    return inner

Maybe(obj.find('thing')) >> computation_1 >> computation_2 >> computation_3...
