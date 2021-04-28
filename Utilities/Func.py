from functools import wraps

"""
New ideas:
    - "Generator functions"
    - For the command-line parsing problem:
        - Simply consume the arguments one-by-one in a generator, yielding a PapisDoc object when a URL is encountered
        - This is (probably) a monad of some sort, but is (probably) not worth abstracting as such
    - Many different situations fit into this general model: (generator) --> Func --> (generator)
        - Map (i.e., something to apply to "mappable")
        - Promise / Observable
        - Operator
        - Transformation
        - Aggregator
        - Class hierarchy?
    - The "map over" class function from the original `Func` class is just composition and currying with a 2-ary generator function (i.e., generator "stacking" / "nesting")
    - Is "generator" itself a monad?
        - Perhaps "finite generator"
        - Its "bind" / "return" could be composed with other functions programmatically to create other IO type containers (such as CLI parsers)
        - Could this be used (abused?) to create semi-arbitrary monads?
    - Operator overloading
    - One way of looking at the "spread / unspread" problem:
        - This is just about containerization---do we want a function to act on a container (monad) instance, or directly on the container's elements (i.e., "inside" the monad)?
"""
class Func:
    """
    A class to represent composable functions of a single variable.

    Functions of many variables can be represented by passing arguments as an iterable.
    The wrapped function's __name__ is assigned to the instance for ease of debugging.

    Attributes
    ----------
    Methods
    -------
    """
    @staticmethod
    def Print(wrapped_fn=None, start=25*'-', end=25*'-', info=None):
        info = info or wrapped_fn.__name__
        def wrapper(*args):
            print(start)
            if info is not None:
                print(info)
            for arg in args:
                print(arg)
            print(end)
            return args
        ret = Func(wrapper)
        return ret if not callable(wrapped_fn) else ret.compose_left(Func.unspread(wrapped_fn))
    @staticmethod
    def Const(const):
        return Func(lambda arg: const)
    @staticmethod
    def Id():
        return Func(lambda arg: arg)
    @staticmethod
    def wrap_take_first(fn):
        return Func(Func.take_first(Func.take_first(fn)))
    @staticmethod
    def take_first(fn):
        @wraps(fn)
        def wrapped(it):
            return fn(list(it)[0]) if any(it) else fn()
        return wrapped
    @staticmethod
    def wrap_unspread(fn):
        return Func(Func.unspread(Func.unspread(fn)))
    @staticmethod
    def unspread(fn):
        @wraps(fn)
        def wrapped(arg):
            return fn(*arg)
        return wrapped
    def print(self, start=25*'-', end=25*'-'):
        self.compose_left(Func.Print(start, end)).compose_right(Func.Print(start, end))
        return self
    def __init__(self, fn):
        """__init__.
        Func constructor.

        Assumes a unital function (i.e. "one in, one out")
        Assigns the wrapped function's name to the instance

        :param fn: Function to wrap
        """
        self._fns = [fn]
        self.__name__ = fn.__name__
    def map_args(self, mp, eval=True):
        """map_args.
        Composes a Func with a 'map' invocation to the right (i.e., acting on the single argument, which is assumed to be iterable)
        TODO: Require `mp` to be a Func instance?

        :param mp: The Func to map over the invoking instance's arg(s)
        :param eval: A Boolean indicating whether the `list` constructor should be applied before passing to the invoking Func
        """
        ret = self.compose_right(lambda args: map(mp, args))
        return ret.compose_left(list) if eval else ret
    def compose_right(self, mp):
        """compose_right.
        Composes with another Func from the left (i.e, f.compose_right(g)(arg) = f(g(arg)))

        :param mp: Func instance to compose with
        """
        self._fns.insert(0, mp)
        return self
    def compose_left(self, mp):
        """compose_left.
        Composes a Func with another Func from the right (i.e., f.compose_left(g)(arg) = g(f(arg)))

        :param mp: Function to compose with
        """
        self._fns.append(mp)
        return self
    # def parallel_with(self, *others):
        # return Func(lambda *args: [self(*args), *[other(*args) for other in others]])
    # def spread(self):
        # self.compose_right(Func.Id())
        # return self
    # def gather(self):
        # return Func.unspread(self)
    # def iterate_over(self, generator):
        # if not callable(generator):
            # generator = Func.Const(generator)
        # @wraps(self)
        # def wrapped(args):
            # ret = []
            # items = generator(args)
            # while True:
                # try:
                    # ret.append(self(next(items)))
                # except StopIteration:
                    # break
            # return ret
        # return Func(wrapped)
    def map_into(self, mappable):
        pass
    def map_onto(self, mappable):
        pass
    def __call__(self, *args):
        for f in self._fns:
            args = f(args)
        return args

from collections import namedtuple
Papis_Opts = namedtuple('Opts', ['tags', 'attrs'])
class Opts(Papis_Opts):
    @staticmethod
    def from_tags(*tags):
        return Opts(set(tags), {})
    @staticmethod
    def from_attrs(**attrs):
        return Opts(set(), attrs)

Opts.from_tags = lambda *tags: Opts(set(tags), {})
Opts.from_attrs = lambda **attrs: Opts(set(), attrs)

split_tags = Func(Func.unspread(str.split)).compose_left(set)\
    .iterate_over(Func.unspread(iter)).compose_left(Func.unspread(set.union))

add_tags = Func(Func.unspread(Func.Print(wrapped_fn=set.update, info='Set update:')))\
    .map_args(Func.Print(wrapped_fn=split_tags, info='Split tags:'), eval=False)\
    .map_args(Func.Print(wrapped_fn=lambda opts: opts.tags, info='Project opts:').gather(), eval=False)

test_tags = {'a', 'b c', 'd e f'}
test_update = {'g h', 'i'}
test_opts = Opts({'boopy', 'shadoopy boo'}, {'title': 'awesomesauce'})

split_tags('a b c')
split_tags(test_tags)
list(Func.unspread(iter)([test_tags]))

Func.wrap_unspread(set.update).print().map_args(split_tags).print()(test_tags, test_update)

test_opts.tags
add_tags(test_opts, Opts.from_tags('yay!'))
add_tags(test_tags, {1, 2, 3})
add_tags._fns

test_opts.attrs
gen_fn = Func(lambda old, new: iter(new.attrs.keys()))
