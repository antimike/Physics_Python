# This is necessary because Sage has trouble telling when expressions involving Hankel fns are equal
# Obviously this works best for functions of r alone
# The next testing fn can be used more generally but depends on `integral_coord_region`
# TODO: Use `expression` class's `test_relation`
def test_equality_radial(qty1, qty2, r_min=0, r_max=10):
    """test_equality_radial.
    Tests equality of two expressions by integrating the absolute square of their difference radially.

    :param qty1: First expression to test
    :param qty2: Second expression to test
    """
    exprs = [complex_norm(qty1 - qty2)(pt_sph()), 1 - complex_norm(qty1/qty2)(pt_sph())]
    ret = None
    while not ret and len(exprs) > 0:
        try:
            ret = integral(exprs.pop(), r, r_min, r_max).subs(k==1).is_trivial_zero()
        except:
            continue
    if ret is None:
        raise ValueError("Evaluation of the comparison integrals failed for the quantities %s and %s" \
            %(qty1, qty2))
    return ret

def test_equality_general(qty1, qty2, bounds={r: [0, 10], th: th, ph: ph}):
    """test_equality_general.
    Tests equality of two expressions by integrating the absolute square of their difference over a supplied region.

    :param qty1: First expression to test
    :param qty2: Second expression to test
    """
    exprs = [complex_norm(qty1 - qty2), 1 - complex_norm(qty1/qty2)]
    ret = False
    while not ret and len(exprs) > 0:
        try:
            ret = integral_coord_region(exprs.pop(), bounds).subs(k==1).is_trivial_zero()
        except:
            continue
    if ret is None:
        raise ValueError("Evaluation of the comparison integrals failed for the quantities %s and %s" \
            %(qty1, qty2))
    return ret

def debug(alias='dbg', **kwargs):
    """debug.
    Decorator that returns a callable instance of class Debugger.
    The basic idea is that the returned instance will be injected into itself as a parameter whose name is provided by the 'alias' parameter.  For example, if alias='dbg', the decorated function will have access to a parameter named 'dbg', which will in turn have access to all class and instance methodsd of the underlying Debugger object.

    :param alias: Name of the parameter to use for injecting the Debugger instance into the function call.
    :param kwargs: Parameters to pass to the Debugger constructor (e.g., names and values of debug template strings, state variables, state reducers, etc.)
    """
    def decorate(fn):
        return Debugger(fn, alias=alias, **kwargs)
    return decorate

def pass_when(predicate, default_func=None, default_val=None):
    """pass_when.
    Decorator intended to "neuter" a class method given a certain (boolean) condition.
    The use-case it was designed for is in the Debugger class:  When an instance's 'debug' property is 'False', decorated methods will effectively become no-ops.
    Needs significant refactoring to be useful in a wider context.

    :param predicate: Boolean function of the decorated function's first argument, which is assumed to be a class instance (i.e., this function is designed to be applied to class methods).  Can be non-callable as well, in which case it is simply evaluated for truthiness.  If 'True' or truthy, the decorated function will simply return a default.
    :param default_func: Function of the decorated function's first argument, which as above is assumed to be an instance of a class.  Should return a default, which will be returned by the decorated function if the 'predicate' is truthy.  Unlike 'predicate', this argument is assumed callable.  This is because frequently, even in cases where a static (i.e., non-parameterizable) default is desirable, that default will *itself* be callable, rendering the kind of simple logic used around the 'predicate' buggy.
    :param default_val: An alternative to default_func which provides a static (non-parameterizable) default value to return if 'predicate' is truthy.  If both arguments are provided, default_val takes precedence and no exception is thrown.
    """
    def decorate(fn):
        @sage_wraps(fn)
        def wrapped(*args, **kwargs):
            obj = args[0] if len(args) else None
            truthy = (callable(predicate) and predicate(obj)) or (not callable(predicate) and predicate)
            if truthy:
                if default_val:
                    return default_val
                elif callable(default_func):
                    return default_func(obj)
                else:
                    return default_val
            else:
                return fn(*args, **kwargs)
        return wrapped
    return decorate

class Debugger:
    """Debugger."""

    _pass_predicate = lambda obj: not obj.active
    _pass_default = lambda obj: obj
    _active_ = None
    _props = {
        'assert_equals': '{} = {}'
    }
    _state = {
        'indent': 0,
        'prefix': '',
        'tab': '\t'
    }
    @staticmethod
    def activate(debugger):
        Debugger._active_ = debugger
        debugger.active = True
    @staticmethod
    def deactivate(debugger):
        Debugger._active_ = None
        debugger.active = False
    @staticmethod
    def debug(*args, **kwargs):
        if Debugger._active_ is None:
            pass
        else:
            try:
                Debugger._active_._debug(*args, **kwargs)
            except:
                pass
    def _reporter(self, template):
        @pass_when(Debugger._pass_predicate, default_func=Debugger._pass_default)
        def wrapped(obj, *args, **kwargs):
            obj._update_state(**kwargs)
            obj._print(template.format(*args, **self.state))
            return obj
        return lambda *args, **kwargs: wrapped(self, *args, **kwargs)
    def _reducer(self, prop_name, func=None):
        @pass_when(Debugger._pass_predicate, default_func=Debugger._pass_default)
        def fixed_func_reducer(obj, _prop_name, *args, **kwargs):
            update = kwargs | {_prop_name: func(self._get_state_variable(_prop_name), *args, **kwargs)}
            obj._update_state(**update)
            return obj
        @pass_when(Debugger._pass_predicate, default_func=Debugger._pass_default)
        def variable_func_reducer(obj, *args, **kwargs):
            update = {}
            obj._update_state(**kwargs)
            if len(args) and (first := args[0] or kwargs.get('func')) and callable(first):
                update[prop_name] = first(obj._get_state_variable(prop_name), *args[1:], **kwargs)
            elif len(args) > 0:
                update[prop_name] = first
            else:
                obj.print_state(prop_name)
            obj._update_state(**update)
            return obj
        ret = fixed_func_reducer if callable(func) else variable_func_reducer
        return lambda *args, **kwargs: ret(self, *args, **kwargs)
    def _update_state(self, **kwargs):
        self._state |= kwargs
    def _set_help_info(self):
        self.__doc__ = self._wrapped.__doc__
        self.__name__ = self._wrapped.__name__
    def __init__(self, fn, **props):
        if fn is None:
            fn = lambda *args, **kwargs: None
        self._alias = props.get('alias')
        self.active = False
        self._wrapped = fn
        self._props |= props
        self._create_debug_methods()
        self._create_state_methods()
        self._set_help_info()
    @pass_when(_pass_predicate, default_func=_pass_default)
    def print_state(self, name):
        self.assert_equals(name, self._get_state_variable(name))
        return self
    def _create_debug_methods(self):
        for name, prop in self._props.items():
            if isinstance(prop, str):
                self.__dict__[name] = self._reporter(prop)
            elif callable(prop):
                self.__dict__[name] = self._reducer(name, func=prop)
            else:
                self._state[name] = prop
    def _create_state_methods(self):
        for name, val in self._state.items():
            self.__dict__[name] = self._reducer(name)
    @property
    def state(self):
        return {name: self._get_state_variable(name) for name in self._state.keys()}
    def _get_state_variable(self, name):
        possible_thunk = self._state.get(name)
        if callable(possible_thunk):
            try:
                return possible_thunk(**self._state)
            except:
                return possible_thunk(self._state)
        elif (redirect := self._state.get(possible_thunk)) and not self._state[possible_thunk] == name:
            return self._get_state_variable(possible_thunk)
        else:
            return possible_thunk
    @property
    def _indent(self):
        return self._get_state_variable('indent')
    @property
    def _tab(self):
        return self._get_state_variable('tab')
    @property
    def _prefix(self):
        return self._get_state_variable('prefix')
    def _print(self, *args):
        print(*[self._indent*self._tab + self._prefix
               + str(arg) for arg in args])
    def _debug(self, *args):
        if len(args) > 0:
            if callable(prop := self.__dict__.get(arg[0])):
                prop(*args[1:])
            elif prop:
                self.print_state(prop)
            elif isinstance(arg[0], str):
                self._print(arg[0].format(*args[1:]))
            elif (var := self._state.get(arg[0])):
                self.print_state(var)
            else:
                pass
        else:
            pass
    def __call__(self, *args, debug=False, **kwargs):
        if not self.active:
            if debug:
                Debugger.activate(self)
            kwargs |= {self._alias: self}
            ret = self._wrapped(*args, **kwargs)
            Debugger.deactivate(self)
            return ret
        else:
            pass
