def map_args(args_fn, kwargs_fn):
    def decorator(fn):
        def wrapper(*args, **kwargs):
            return fn(*args_fn(*args, **kwargs), **kwargs_fn(*args, **kwargs))
        return wrapper
    return decorator

# Guiding principle: Transformations of the 'kwargs' should depend only on the instance vars and the original kwargs
# (i.e., not on positional 'args')
def method_decorator(instance_args_fn, instance_kwargs_fn):
    return map_args(
        lambda instance, *args, **kwargs: instance_args_fn(instance, *args, **kwargs),
        lambda instance, *args, **kwargs: instance_kwargs_fn(instance, **kwargs)
    )

def get_args(*args, **kwargs):
    return args

def get_kwargs(*args, **kwargs):
    return kwargs

def tail(*args):
    return args[1:]

def compose(f, g, *args):
    return f(*g(*args))

# Wraps a function guaranteed to return either True or False on all inputs
# Swallows all exceptions silently, so potentially dangerous
class Boolean:
    def __init__(self, describes=None):
        self._describes = describes
    def describes(self, val):
        try:
            if self._describes(val):
                return True
            return False
        except:
            return False
    # TODO: Add exception handling for 'other' calls?
    def __and__(self, other):
        return Boolean(lambda x: self._describes(x) and other._describes(x))
    def __or__(self, other):
        return Boolean(lambda x: self._describes(x) or other._describes(x))
    def __xor__(self, other):
        return Boolean(lambda x: (self & ~other)|(~self & other))
    def __invert__(self):
        return Boolean(lambda x: not self._describes(x))
    # def implies(self, other):

class Proposition:
    def __init__(self, quantifiers):
        pass


b = Boolean(lambda x: True)

class Set:
    # I could implement some error checking with a try/except,
    # but this is better left to the implementer of 'contains'
    def __init__(self, contains):
        if not isinstance(contains, Boolean):
            raise TypeError
        self._contains = contains
    def contains(self, val):
        return self._contains.describes(val)
    all_sets = Set(Boolean(lambda x: isinstance(x, Set)))
    @staticmethod
    def union(first, second):

class Function:
    def __init__(self, f, domain, codomain):
        self._func = f
        self._domain = domain
        self._codomain = codomain
    def maps(self, val):
        return self._domain(val)
    def maps_to(self, val):
        return self._codomain(val)

class Pure_Function(Function):
    def __init__(self, f):
        super(self, f)

def method_transformation(instance_args_fn):
    return method_decorator(instance_args_fn, get_kwargs)

def method_options_filter(instance_kwargs_fn):
    return method_decorator(
        lambda instance, *args, **kwargs: args,
        instance_kwargs_fn
    )

get_opts = method_decorator(
    lambda instance, *args, **kwargs: args,
    lambda instance, **kwargs: {**instance._opts, **kwargs}
)
