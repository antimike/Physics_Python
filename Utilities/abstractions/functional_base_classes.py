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
    # @staticmethod
    # def union(first, second):

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

## Decorator cemetery

get_opts = method_decorator(
    lambda instance, *args, **kwargs: args,
    lambda instance, **kwargs: {**instance._opts, **kwargs}
)

def apply_defaults(fn):
    def wrapper(obj, *args, **kwargs):
        return fn(obj, *args, **{**obj._opts, **kwargs})
    return wrapper

@bind_args(compose_reducers)
def bind_kwargs(*reducers):
    # reducer = lambda instance, reducers, kwargs: 
    def decorator(fn):
        def wrapper(instance, *args, **kwargs):
            return fn(instance, *args, reducer(instance, kwargs))
        return wrapper
    return decorator

# Related classes:
# - State
# - Hamiltonian_State
class Observable:
    def __init__(self, manifold):
        self._manifold = states
    @property
    def manifold(self):
    # The set from which this observable takes values.
        return self._manifold

# For now, doesn't expose anything fancy like "contra-" or "covariant" Booleans, or type-checks
# Eventual goal: wire up support for explicit Domain, Codomain, etc. to allow more rigorous type-checking
# Possibly implement an unchecked base class and a more "rigorous" derived class
# For checking functional properties like commutativity, we need to make some assumptions about the algebraic structures of the Domain and Codomain, so this will have to be limited to derived classes like Linear_Reducer (e.g.)
# Additional possibilities: 
# - pair (member function that provides an explicit construction to pair a state with an update into a properly-formed argument for the reducer)
# - get_state, get_update (projection functions; see previous)
# - "Passive" and "Active" derived classes, depending on whether the logic of composition is implemented through modified projectors (i.e., modified Domain/Codomain) or a modified internal reducer
class Reducer:
    def __init__(self, fn):
        self._reducer = fn
        self._get_state = lambda pair: pair[0]
        self._get_update = lambda pair: pair[1]
        self._pair = lambda state, update: (state, update)
        self._adjoint = Reducer(lambda state, update: self._reducer(update, state))
    # def reduce(contravariant, covariant):
    def reduce(state, update):
        return self._reducer(state, update)
    # TODO: Replace with Haskell-like point-free implementation based on functional "pair"
    @property
    def adjoint(self):
        return self._adjoint
    def left_pseudolift(self):
        return lambda pair: self._pair(self._reducer(**pair), self._get_state(pair))
    def right_pseudolift(self):
        pass
    def forgetful_pseudolift(self):
        pass
    def compose_eager(self, second):
        return lambda state, update: second(self._reducer(state, update), update)
    def compose_deferred(self, second):
        return lambda state, update: second(state, self._reducer(state, update))
    def compose_forgetful(self, second):
        return lambda state, update: second(update, self._reducer(state, update))
    def __add__(first, second):
        pass

# A decorator that functions as a container for reducers
class Redecorator:
    def __init__(self, *reducers):
        self._reducer = sum(reducers)

# A reducer f: (S, S) -> S can be "lifted" to a function (S, S) -> (S, S) in three obvious ways.
# Two ways are "conservative:" (S_1, S_2) |-> (S_1, r(S_1, S_2)) and the adjoint, each of which fixes one of the subdomains.
# The difference between these two can be thought of as "immediate state update" (right arg fixed, left mutable) and "accumulated state update deferred" (left arg fixed representing the state to be updated; right arg mutable)
# The other is "forgetful:" (S_1, S_2) |-> (S_2, r(S_1, S_2))
# If the state space is itself structured in some algebraic way, and if reducers are assumed to respect this structure, then under certain conditions these three possibilities might constitute some sort of "basis" for the set of *all* possible ways of lifting a reducer to a function (S, S) -> S.

# This can all be made sense of if we think of reducers as contravariant or covariant, like tensors but without the assumed linear structure.
# Essentially, the pair (S_1, S_1) splits into "state vectors" (tangent to some manifold or generalization thereof, perhaps) and "update forms" (cotangents).
# The type of the return value of r determines its classification.

# Of course, the specification of any pair of covariant and contravariant reducers would be enough to construct the entire future "update trajectory" given an initial seed state and a single update event.
# Thus, this model might be too idealistic.
# Instead, we should think about "half-covariant" reducers---reducers which return future updates with some degree of uncertainty or imprecision---being paired with fully contravariant reducers used to actually update the internal state as events unfold.
# Or perhaps we should think of the state itself acquiring "fuzziness" or uncertainty over time, in which case the contravariant reducers are also in reality only "partly" or "imperfectly" covariant.

# Analogy to physics: Anomalous circumstances in which an "update" (impulse) fails to result in the usual displacement (application of reducer) can be interpreted as the result of:
# - noninertial frame
# - the presence of some external force
# - inertia?

# "Discrete quantization" of a reduction framework...
# ...would be to imagine a superposition of class instances as being the "state" to be updated
# ...or could be a superposition of states ("fuzzy reducers") for a fixed class instance, depending on the interpretation of the class.

# How best to represent reducers in class form, in Python?
"""
'Reducer' should expose:
* reduce
* left-semilift
* right-semilift
* skew-semilift
* contra-/co-variance
"""

class Class_Decorator:
    def __init__(self, *args, **kwargs):
        pass
    def __call__(arg):
        pass
    def __getattribute__(self, name):
        try:
            pass
        except AttributeError:
            pass
        else:
            pass

class Data_Context:
    def __get__(self, obj, objtype=None):
        raise NotImplementedError
    def __set__(self, obj, val):
        raise NotImplementedError
    def __set_name__(self, owner, name):
        raise NotImplementedError


class View(ABC):
    pass

class Serializable(ABC):
    pass

class Model(ABC):
    pass




@bind_method_args
@contextual
class Element:
    @classmethod
    @cached_property




def right_compose_reducers(reducers):
    def reducer(first, second):
        for r in reducers:


def bind_args(reducer):
    def decorator(fn):
        def wrapper(instance, *args, **kwargs):
            return fn(instance, *reducer(instance, args), kwargs)
        return wrapper
    return decorator

def reduce_data_opts(table, opts):
    return {**table._opts, **opts}

def reduce_col_title_opts(table, opts):
    data_kwargs = reduce_data_opts(table, opts)
    col_title_opts = {**table._opts['col_title_opts'], **opts, **opts['col_title_opts']}
    return {**reduce_data_opts(table, opts), 'col_title_opts': col_title_opts}

