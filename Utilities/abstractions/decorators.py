def filter_args(predicate):
    """Filters args based on predicate
    Removes arg when predicate is failed"""
    def decorator(fn):
        def wrapper(*args, **kwargs):
            return fn(*filter(predicate, args), **kwargs)
        return wrapper
    return decorator

def replace_args(predicate, val):
    """Filters *args based on predicate, replacing arg with 
    default value when predicate fails"""
    def decorator(fn):
        def wrapper(*args, **kwargs):
            return fn(
                *map(lambda x: x if predicate(x) else val, args),
                **kwargs
            )
        return wrapper
    return decorator

def sublist_specifier():
    pass

# TODO: Reimplement in terms of iterators
def take(collection, n):
    """Takes the first n elements of a collection"""
    return collection[:n]

# unweaving_func::Boolean
# [x if test(i) 

def scramble(predicate, args):
    selected = [x for i, x in enumerate(args) if predicate(i)]
    unselected = [x for i, x in enumerate(args) if not predicate(i)]
    return [*selected, *unselected]


def unscramble(predicate, beep_boop, debug=False):
    args = beep_boop
    args.reverse()
    if debug:
        print('-----' + str(args))
    args = scramble(predicate, args)
    if debug:
        print('-----' + str(args))
    args.reverse()
    if debug:
        print('-----' + str(args))
    return args


def test_unscramble(q, predicate, debug=False, max_iterations=100):
    print(q)
    qs = scramble(predicate, q)
    iterations = 1
    while (not qs == q) and iterations < min(len(q), max_iterations):
        print(qs)
        qs = unscramble(predicate, qs, debug=debug)
        iterations += 1
    if qs == q:
        print('It took {:d} iteration(s) to reconstruct the original list.'.format(iterations))
    else:
        print('Failed to reconstruct the original list.  Ran for {:d} iterations.'.format(iterations))
        print('Final state:')
        print('{:s}'.format(str(qs)))



test_unscramble([*range(5)], lambda x: x == 0 or x == 3, debug=True)

def test_slice(total_length, slice_length):
    q = [*range(total_length)]
    fn = lambda x: x >= slice_length
    test_unscramble(q, fn, debug=True)


test_fns = {
    'modular square': lambda x: x**2%27 < 10,
    'even': lambda x: x%2 == 0,
    'divisible by 7': lambda x: x%7 == 0,
    'prime': lambda x: len([y for y in range(2, x - 1) if x%y == 0]) == 0
}

for key, fn in test_fns.items():
    print('', '------------------------------------------')
    print('Testing predicate `{:s}`:'.format(key))
    test_unscramble([*range(28)], fn, debug=True, max_iterations=2)

"""
How to invert a boolean sort (subset selection) of a list:
first = [x for test(x) and x in l]
second = [x for not test(x) and x in l]
reconstructed = first + second
original = (reverse + split(test) + reverse)(reconstructed)
"""

identity = lambda x: x
none_check = lambda default, proj=identity: replace_args(lambda x: x is not None, default)

@none_check(identity)
def compose_args(f, g, **kwargs):
    """Composes single-valence functions f and g, sharing kwargs
    Handles `None` as `identity`"""
    return lambda x: f(g(x, **kwargs), **kwargs)

@none_check(identity, lambda args: args[:2])
def compose(f, g, args_factory, kwargs_factory):
    """Offers a general way to 'compose' functions with arbitrary valences"""
    def ret(*args, **kwargs):
        inner = g(*args, **kwargs)
        return f(*args_factory(inner, *args, **kwargs),
                 **kwargs_factory(inner, *args, **kwargs))

@none_check(0)
def test(x, y, **kwargs):
    return x**2 - y**2

assert(test(None, None) == 0 and test(None, 2) == -4)

# compose_args = lambda f, g: compose(f, g, lambda inner, *args, **kwargs

def fold_right(l, accumulator, seed=None):
    """Folds a list from the right with initial value `seed`
    Accumulator is expected to handle `None` gracefully"""
    while len(l) > 0:
        seed = accumulator(l[-1], seed)
        l = l[:-1]
    return seed

def fold_args(combinator, seed=None):
    """Right-folds passed arguments into a single arg based on combinator"""
    def decorator(fn):
        def wrapper(*args, **kwargs):
            return fn(fold_right(*args, combinator, seed), **kwargs)
        return wrapper
    return decorator

@filter_args(lambda x: x is not None)
@fold_args(compose)
def bind_kwargs(reducer):
    def decorator(fn):
        def wrapper(*args, **kwargs):
            return fn(*args, **reducer(args, kwargs))
        return wrapper
    return decorator

@filter_args(lambda x: x is not None)
@fold_args(compose)
def bind_args(reducer):
    def decorator(fn):
        def wrapper(*args, **kwargs):
            return fn(*reducer(args), **kwargs)
        return wrapper
    return decorator

def none_check(default_val, *vals):
    return [v if v is None else default_val for v in vals]




def decorate_methods(*decorators):
    """A class decorator which applies the passed list of
    method decorators to every instance method."""
    total = lambda fn: fn
    while len(decorators) > 0:
        last = decorators[-1]
        total = lambda fn: last(total(fn))
    def class_decorator(Cls):
        class Cls_Wrapper(object):
            def __init__(self, *args, **kwargs):
                self._inner_instance = Cls(*args, **kwargs)
            def __getattribute__(self, s):
                try:
                    ret = super(Cls_Wrapper, self).__getattribute__(s)
                except AttributeError:
                    pass
                else:
                    return ret
                ret = self._inner_instance.__getattribute__(s)
                if type(ret) == type(self.__init__):
                    return total(ret)
                else:
                    return ret
        return Cls_Wrapper
    return class_decorator


# A reducer f: (S, S) -> S can be "lifted" to a function (S, S) -> (S, S) in three ways.
# Two ways are "conservative:" (S_1, S_2) |-> (S_1, r(S_1, S_2)) and the adjoint, each of which fixes one of the subdomains.
# The difference between these two can be thought of as "immediate state update" (right arg fixed, left mutable) and "accumulated state update deferred" (left arg fixed representing the state to be updated; right arg mutable)
# The other is "forgetful:" (S_1, S_2) |-> (S_2, r(S_1, S_2))
