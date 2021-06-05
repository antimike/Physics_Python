from functools import wraps
import pickle
from collections import namedtuple
import logging
from inspect import signature

logger = logging.getLogger('Picklebarrel')
# log_messages = {
    # 'stewing': "

def get_public_methods(cls):
    return [
        obj for obj in dir(cls) if callable(getattr(cls, obj)) and not obj.startswith('__')
    ]

def decorate_class_methods(cls, decorator_factory, **kwargs):
    for method in get_public_methods(cls):
        setattr(cls, method, decorator_factory(method, **kwargs.get(method, {})))

def log_function_calls(
    log: logging.Logger = None,
    enter_msg: str =
    """Calling function '%(__name__)s' with parameters %(args)s and keywords %(kwargs)s""",
    exit_msg: str =
    """Successfully completed call to function '%(__name__)s' with parameters %(args)s, keywords %(kwargs)s, and return value %(return)s""",
    exception_msg: str =
    """Call to function '%(__name__)s' failed due to raised exception of type '%(exception_type)s'""",
    bind_failure_msg: str =
    """Failed to bind arguments %(args)s and keywords %(kwargs)s to function '%(__name__)s'""",
    enter_level: logging=logging.INFO,
    exit_level=logging.INFO,
    exception_level=logging.ERROR
):
    @wraps(_log_function_calls)
    def decorator(fn):
        return _log_function_calls(fn, **locals())
    return decorator

def _log_function_calls(fn, **kwargs):
    """
    Decorator
    """
    keywords = signature(log_function_calls)
    log = log or logging.getLogger(fn.__module__)
    sig = signature(fn)
    name = fn.__name__

    @wraps(fn)
    def wrapped(*args, **kwargs):
        params = {'args': args, 'kwargs': kwargs, '__name__': name}

        try:
            params |= sig.bind(*args, **kwargs).arguments
        except TypeError:
            log.log(exception_level, bind_failure_msg % params)
            raise

        log.log(enter_level, enter_msg % params)

        try:
            result = fn(*args, **kwargs)
        except Exception as e:
            log.log(exception_level, exception_msg % params, ex_info=1)
            raise

        log.log(exit_level, exit_msg % params)
        return result
    return wrapped

class Picklebarrel:
    """
    Convenience class to manage pickling of data computed over a range of inputs

    Public methods
    ==============
    stew
        :param fn: Unary function whose values will eventually be pickled
        :param args_list: List of args for which outputs will be computed and pickled
    ferment
    preserve
        :param fname: Filename to write data (pickle) data to
    picklefunc (decorator factory)
        :param args_list: List of args for which outputs will be computed and pickled

    >>> barrel = Picklebarrel()
    >>>
    >>> @barrel.picklefunc(range(5))
    >>> def test_func(arg):
    >>>   return arg**2
    >>>
    >>> type(barrel._fns['test_func'])
    <class 'zip'>
    >>> barrel.ferment()
    >>> barrel._fns
    {'test_func': {0: 0, 1: 1, 2: 4, 3: 9, 4: 16}}
    """

    def __init__(self):
        self._fns = {}
        self._computed = {}

    def stew(self, fn, args_list, name=None, overwrite_existing=False, sort_args=True):
        """Picklebarrel.stew
        Registers a function to be pickled later, whenever self.preserve() is called.
        Stores only generators (lazy) at the time of registration.

        :param fn: Unary function whose values will eventually be pickled
        :param args_list: List of args for which outputs will be computed and pickled
        """
        name = name or fn.__name__
        # if name in self._fns and not overwrite_existing:
            # raise ValueError(
                # "The name '{name}' has already been stewed in this barrel!  Pass a different name or else specify 'overwrite=True'.".format(name=name)
            # )
        args_list += self._fns[name].get('to_compute', [])
        if sort_args:
            args_list.sort()
        if not overwrite_existing:
            args_list = [arg for arg in args_list if arg not in self._computed.get(name)]
        args_list.reverse()
        self._fns[name] = {
            'func': fn,
            'to_compute': args_list
        }

    def retrieve(self, fname, prefix='', overwrite=False):
        with open(fname, 'rb') as file:
            data = pickle.load(file)
        

    def ferment(self, fn=None, args=None):
        """Picklebarrel.ferment
        Evaluates the generators stored internally in self._fns to produce the actual data to be pickled.

        :kwarg fn: Specific function to ferment.  If None, all registered functions are fermented
        :kwarg args: Specific arguments to ferment.  If None, args passed at the time self.stew() was called are fermented
        """
        self._fns = {name: dict(list(pair_gen)) for name, pair_gen in self._fns.items()}

    def preserve(self, fname):
        """Picklebarrel.preserve
        Writes (pickles) the data in self._fns to the given file.
        Data is written in the form of a namedtuple to allow property-style access to function values upon unpickling.

        :param fname: Filename to pass to pickle.dump()
        """
        Data = namedtuple("Data", self._fns.keys())
        data = Data(**self._fns)
        with open(fname, "wb") as file:
            pickle.dump(Data)

    def picklefunc(self, args_list):
        """Picklebarrel.picklefunc
        Decorator factory to register a decorated function with the passed args_list via self.stew()

        :param args_list: List of arguments to call on decorated function
        """

        def decorator(fn):
            self.stew(fn, args_list)

            def wrapped(arg):
                return fn(arg)

            return wrapped

        return decorator
