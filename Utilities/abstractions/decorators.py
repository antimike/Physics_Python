
@bind_args(compose_reducers)
def bind_kwargs(*reducers):
    # reducer = lambda instance, reducers, kwargs: 
    def decorator(fn):
        def wrapper(instance, *args, **kwargs):
            return fn(instance, *args, reducer(instance, kwargs))
        return wrapper
    return decorator

# A reducer f: (S, S) -> S can be "lifted" to a function (S, S) -> (S, S) in three ways.
# Two ways are "conservative:" (S_1, S_2) |-> (S_1, r(S_1, S_2)) and the adjoint, each of which fixes one of the subdomains.
# The difference between these two can be thought of as "immediate state update" (right arg fixed, left mutable) and "accumulated state update deferred" (left arg fixed representing the state to be updated; right arg mutable)
# The other is "forgetful:" (S_1, S_2) |-> (S_2, r(S_1, S_2))

# def deferred

def right_compose_reducers(reducers):
    def reducer(first, second):
        for r in reducers:


def bind_args(reducer):
    def decorator(fn):
        def wrapper(instance, *args, **kwargs):
            return fn(instance, *reducer(instance, args), kwargs)
        return wrapper
    return decorator
