from pint import UnitRegistry
from pint import Quantity

class SageQuantity(Quantity):
  """SageQuantity.
  Subclass of pint's Quantity designed to wrap Sage symbolic expressions.
  Should also work for Manifold-related objects (e.g., scalar- and vector-fields)

  TODO: Implement this!
  - Should add wrapped expression or object's __dict__ to its own
    - i.e., Sage methods should be directly callable
  - Should expose a method `__units__` or `units` for consumption by checkers (decorators, e.g.)
  """
  pass

""" Units """
# TODO: Implement some helpers / decorators here?
