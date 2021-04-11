def class_decorator(fn):
    class _decorator_test:
        def __init__(self, fn):
            self._wrapped = fn
        def __call__(self, *args, **kwargs):
            return self._wrapped(*args, **kwargs, test_var=2)
    return _decorator_test(fn)

@class_decorator
def test_fn(a, b, test_var=None):
    if test_var is None:
        print('Undecorated!')
    else:
        print('Decorated: test_var = {}'.format(test_var))
    return

test_fn(1, 2)
