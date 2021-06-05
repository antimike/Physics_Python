from functools import wraps, cache
from collections import namedtuple
from sage.manifolds.operators import *
from sage.manifolds.catalog import Sphere
from scipy import special as fns
from sympy import factorial2, assoc_legendre
from pint import UnitRegistry
import logging

sys.path.append('/home/user/Documents/Python/Utilities')
import debugger as debg





