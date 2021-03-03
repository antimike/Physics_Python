import abc
import functools as ft
from enum import Enum, auto

class Function:
    def __init__(self, inner_func, **arg_list):
        self._inner_func = inner_func
        self._domain = Domain(args=arg_list)
        # This will be an automatically-constructed "Formal_Boolean" (i.e., not evaluated) unless specified directly
        self._codomain = codomain
    # TODO: Add better error handling (more relevant exceptions, etc.)
    def __call__(self, *args, **kwargs):
        assert(self._domain.contains(*args, **kwargs))
        ret = self._inner(*args, **kwargs)
        assert(self._codomain.contains(ret))
        return ret

class Formal_Boolean(Enum):
    TRUE = auto()
    FALSE = auto()
    UNTESTABLE = auto()
    TESTABLE = auto()


class Entity:
    def get(self, *args, **kwargs):
        pass
    @property
    def dependency_structure(self):
        return self._dependency_structure
    def __init__(self, name, validator, context):
        self._context = context
        self.validator = validator
        self.name = name

class Context:
    def __init__(self, dependencies, sources, entities):
        pass
    def lookup(self, key, prop):
        raise NotImplementedError
    def add_entity(self, name, validator, *required_data, **props):
        setattr(self, name, Entity(name, validator, self))
    def add_service(self, service):
        for method in service.service_methods:
            setattr(self, method, getattr(service, method))

# class Element:


        
        

class Service(abc.ABC):
    @abc.abstractmethod
    @property
    def service_methods(self):
        raise NotImplementedError

class Source:
    pass

class Map:
    pass




"""

# Concrete TODOs:

- Class for callable, addressable functions
    ```
    f = FnWithAttributes(inner_func, *props)
    ```

# Desired pattern:
`_.element("Au").reflectivity(omega_range)`
_ = Context("element", chem=chem, rakie=rakie)
_.add_entity("element")
_.configure_key(element, lambda elem: elem.symbol, ...)
_.element.obtains(blah).from(beep, boop).via(fns) :: Contract
_.element.prop(mass).from(fn, reliability=1).check(units_decorator)
_.element.prop(reflectivity).args(

==> Entities should be both callable (e.g., element("Au")) and addressable

Source priority?
    - Manage implicitly by always returning a dictionary of possible answers to queries?

@requires(data)     # Shouldn't be necessary, but might be a good testing framework
@units(...)
def some_property(self):
    
# IDEA:

Use interfaces / metaclasses to keep track of which functions require which properties?
* Auto-constructed dependency trees

# Nice-to-haves:
* Method to return LaTeX for calculation methods
    * Should be possible in Sage

"""
