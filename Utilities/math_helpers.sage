from collections import namedtuple
from sage.manifolds.operators import *
from sage.manifolds.catalog import Sphere
from scipy import special as fns
from sympy import factorial2

def Y_lm_jackson(l, m):
  """Y_lm_jackson.
  Spherical harmonic, with phase and normalization convention as in Jackson.
  See Jackson 3.53.

  :param l: Order (angular momentum)
  :param m: Order (magnetic)
  """
  return EEE.scalar_field(spherical_harmonic(l, m, th, ph)*(-1)^m)

def L_operator(scalar):
  """L_operator.
  Vector angular momentum operator.
  See Jackson 9.101.

  :param scalar: Scalar field to which operator should be applied.
  """
  return -i*p.cross(scalar.gradient())

def X_lm_jackson(l, m):
  """X_lm_jackson.
  Vector spherical harmonic, defined by X_lm = [l(l+1)]^(-1/2)*LY_lm.
  See Jackson 9.119.

  :param l: Order of the multipole (angular momentum)
  :param m: Order of the multipole (magnetic)
  """
  return 1/sqrt(l*(l+1))*L_operator(Y_lm_jackson(l, m))

def a_lm_E_long_wavelength(l, m, Q):
  """a_lm_E_long_wavelength.
  Compute the multipole coefficients a_lm^E as a function of l, m, and the static moments Q.
  See Jackson 9.169.

  :param l: Order of the multipole (angular momentum)
  :param m: Order of the multipole (magnetic)
  :param Q: Static multipole moment.  Should include the induced multipole moment due to magnetic induction.
  """
  return c*k^(l + 2)/(i*factorial2(2*l + 1))*sqrt((l + 1)/l)*Q

@_catch_NameError
def E_lm_E_long_wavelength(l, m, a, k=k, Z=Z):
  """E_lm_E_long_wavelength.
  Electric field of an electric multipole with coefficient a.
  See Jackson 9.122.

  :param l: Order of the multipole (angular momentum)
  :param m: Order of the multipole (magnetic)
  :param a: Multipole coefficient
  :param k: (Optional) wavevector of radiation.  If not provided, the variable 'k' is used.
  :param Z: (Optional) wave impedance.  If not provided, the variable 'Z' is used.
  """
  return i*Z*a/k \
    *(((l + 1)/(2*l + 1)*k*r*spherical_hankel1(l+1, k*r) \
       - l/(2*l + 1)*k*r*spherical_hankel1(l-1, k*r) + i*spherical_hankel1(l, k*r)) \
      *grad(Y_lm_jackson(l, m)) \
      - i*l*(l+1)/r*spherical_hankel1(l, k*r)*Y_lm_jackson(l, m)*frame[1])

# Vector calculus and tools for working with Sage's scalar_field and vector_field

def conjugate_scalar(scalar):
  """conjugate_scalar.
  Wraps the complex conjugate of a scalar field as another scalar field on the same manifold.

  :param scalar: Scalar field
  """
  manifold = scalar._manifold
  ret = manifold.scalar_field()
  for chart in manifold.atlas():
    try:
      ret.add_expr(conjugate(scalar.coord_function(chart=chart).expr() + 0*i), chart=chart)
    except:
      pass
  return ret

def hermitian_conjugate_vector(vector):
  """hermitian_conjugate_vector.
  Wraps the Hermitian conjugate of a vector field as a differential 1-form on the same manifold.

  :param vector: Vector field to conjugate
  """
  manifold = vector._domain
  ret = manifold.diff_form(1)
  for chart in manifold.atlas():
    for j in range(1, manifold.dim() + 1):
      ret.add_comp(chart.frame())[j] = conjugate(vector.comp(chart.frame())[j].expr() + 0*i)
  return ret

def vector_complex_norm_squared(v):
  """vector_complex_norm_squared.
  Returns the norm-squared of a vector field with complex components, as a scalar field on the same manifold.

  :param v: Vector field to find norm of
  """
  return hermitian_conjugate_vector(v)['_i']*v['i']

@_catch_NameError
def H_lm_E_long_wavelength(l, m, a, E=None, k=k, Z=Z):
  """_H_lm_E_long_wavelength.
  Returns the magnetic multipole field due to electric multipole of given order with given coefficient.
  If optional argument E is provided, computes the result by taking the curl of E.
  See Jackson 9.109.

  :param l: Order of the multipole (angular momentum)
  :param m: Order of the multipole (magnetic)
  :param a: Coefficient of the (electric) multipole
  :param E: (Optional) Electric field of the multipole
  :param k: (Optional) wavevector of radiation.  If not provided, the variable 'k' is used.
  :param Z: (Optional) wave impedance.  If not provided, the variable 'Z' is used.
  """
  if E is None:
    E = E_lm_E_long_wavelength(l, m, a, k=k, Z=Z)
  return -i/(k*Z)*curl(E)

def initialize_EM_variables(subs=None):
  """initialize_EM_variables.
  Initializes variables used in EM computations: c, k, Z, R, V, t, omega, epsilon_o, mu_0.
  Initializes and returns 'substitute_exprs,' a dictionary of substitutions indexed by variable name.  If optional 'subs' argument is provided, new dictionary is appended to it.

  :param subs: (Optional) Dictionary of existing 'substitute_exprs.'
  """
  c,k,Z = var('c k Z_0')
  R,V,t = var('R V t')
  omega, epsilon_0, mu_0 = var('omega epsilon_0 mu_0')
  subs = subs if subs is not None else {}
  subs |= {
    omega: k*c,
    epsilon_0: 1/(Z*c),
    mu_0: Z/c
  }
  return subs

@_catch_NameError
def diff_cross_section_pure(l, m, a):
  """diff_cross_section_pure.
  Returns the differential cross-section of a "pure" multipole.
  See Jackson 9.151.

  :param l: Order of the multipole (angular momentum)
  :param m: Order of the multipole (magnetic)
  :param a: Coefficient of the multipole
  """
  return Z/(2*k^2)*norm(a + 0*i)*vector_complex_norm_squared(X_lm_jackson(l, m))

def full_angular_integral(angular_fn):
  return integral(sin(th)*angular_fn(pt()), th, 0, pi).integral(ph, 0, 2*pi)

def surface_integral_scalar(scalar, bounds):
  manifold = scalar._manifold
  chart, coords = _get_chart(bounds.keys(), manifold)
  vol_element = manifold.volume_form().comp(chart.frame())[1, 2, 3]
  ret = (scalar*vol_element)(manifold.point(coords), chart=chart)
  for var, bound in bounds.items():
    if isinstance(bound, list) or isinstance(bound, tuple):
      ret = integral(ret, var, *bound)
    else:
      ret = ret.subs(var==bound)
  return ret

test_scalar = EEE.scalar_field(x^2 + y^2 + z^2, chart=cart)
surface_integral_scalar(test_scalar, cart_bounds)
cart_bounds = {x: [1, 2], y: [-2, 2], z: 4}
set(cart_bounds.keys())

def volume_integral():
  pass

def flux_integral():
  pass



"""_Q_lm_surface_charge(1, 0, spherical_surface_charge(scalar_potential_jackson),
                     [(th, 0, pi), (ph, 0, 2*pi)], [r==R])
_Q_lm_radiative(1, 0, spherical_surface_charge(scalar_potential_jackson))"""
def _Q_lm_surface_charge(l, m, sigma, ranges, subs, vol_elem=r^2*sin(th)):
  """_Q_lm_surface_charge.

  :param l:
  :param m:
  :param sigma:
  :param ranges:
  :param subs:
  :param vol_elem:
  """
  res = (vol_elem*sigma*r^l*conjugate_scalar(Y_lm_jackson(l, m)))(pt()).subs(subs)
  for rg in ranges:
    res = integral(res, rg)
  return res.subs(subs)

def scalar_potential(A, B):
  """scalar_potential.

  :param A:
  :param B:
  """
  return EEE.scalar_field(sum(
    ((a*r^l + b/r^(l + 1))*legendre_P(l, cos(th)) for l, (a, b) in enumerate(zip(A, B)))
  ))

# Helper functions (internal)

def _catch_NameError(fn):
  def wrapper(*args, **kwargs):
    try:
      fn(*args, **kwargs)
    except NameError as e:
      import sys
      raise NameError(str(e) + ' Perhaps try calling math_helpers.initialize_EM_variables()?').with_traceback(sys.exec_info()[2])

def print_msgs(*strings, prefix='\t', level=0, abort=True):
  if abort:
    return
  for string in strings:
    print(prefix*level + string)

#debug_msg(*strings, 

def _debug(func_name='', msgs={}):
  _debugger = Debugger(func_name, msgs)
  def __debug(fn):
    def wrapped(*args, **kwargs, debug=False, debug_level=0):
      _debugger.enter(debug=debug, level=level, args=args, kwargs=kwargs)
      ret = fn(*args, **kwargs, debugger=_debugger)
      _debugger.end(ret)
    return wrapped
  return __debug(fn)

class Debugger():
  def _out(self, **kwargs):
    for msg, params in kwargs.items():
      msg = self._debug_msgs.get(msg)
      if msg is None:
        print('Debug message with key {} not found!'.format(str(msg)))
      elif callable(msg):
        try:
          print(msg(params))
        except:
          print('Your debug message "{}" failed to print with params "{}"'.format(str(msg), str(params)))
      else:
        print(str(msg) + str(params))

  @property
  def context(self):
    return self._debug_action if self._debug else self._do_nothing
  def _prepare_state(self, args, kwargs, debug_level):
    if self._debug:
      self._msg_count = 0
  def enter(self, debug=False, debug_level=0, args=None, kwargs=None):
    self._debug = debug
    self._debug_level = debug_level
    self._prepare_state(args, kwargs, debug_level)
  def end(self, return_val):
    self._print_msg('return')
  def __init__(self, func_name, **kwargs):
    self._func_name = func_name
    self._debug = False
    self._actions = kwargs
  #def _parse_kwargs(self, kwargs):
    #for term in kwargs:
  def _perform_debug_action(self, action):
    if not self._debug:
      return

format_string = '{} {}'*var
format_string.format('boopy', 'shadoopy')

@debug(out='{} {}', recurse={'increment': lambda x: x + 1, 'decrement': lambda x: x + 1})
state(recursion_depth.increment())
log()
pause()
#trigger.recurse(
with dbg.

# Dot-chaining!

{
  'break': '-'*25,
  'level': lambda l: 'Level of recursion: {}'.format(l),
  'State': lambda o, u, _u: 'State: ordered = {}, unordered = {}, _unordered = {}'.format(o, u, _u),
  'Testing': lambda o, u: 'Testing: ordered = {}, unordered = {}'.format(o, u),
  '

def _get_chart(coords, manifold, debugger=None):
  level = 0
  def _get_chart_recursive(ordered, unordered):
    level = len(ordered)
    _unordered = set(unordered)
    ret = None
    while ret is None and unordered:
      first_arg = ordered + [coord := unordered.pop()]
      second_arg = _unordered.difference({coord})
      print_msgs('-'*25+'', 'Level of recursion: {}'.format(len(ordered)),
            'State: ordered = {}, unordered = {}, _unordered = {}'.format(ordered, unordered, _unordered),
            'Testing: ordered = {}, unordered = {}'.format(first_arg, second_arg),
            '-'*25+'', level=level, abort=not debug)
      ret = _test(
        first_arg,
        second_arg)
      print_msgs('-'*25+'', 'Returning to level: {}'.format(len(ordered)),
            'Found: {}'.format(str(ret)),
            '-'*25+'', level=level, abort=not debug)
    return ret
  def _test(ordered, unordered):
    if len(unordered) == 0:
      try:
        string = ' '.join(map(str, ordered)).strip()
        print_msgs('Testing {} from ordering {}'.format(string, str(ordered)), level=level, abort=not debug)
        ret = (manifold.get_chart(string), ordered)
        print_msgs('Found: {}'.format(str(ret)), level=level, abort=not debug)
      except KeyError:
        print_msgs('Not found!', level=level, abort=not debug)
        return None
    else:
      print_msgs('Received ordered = {}, unordered = {}; recursing further'.format(str(ordered), str(unordered)), level=level, abort=not debug)
      return _get_chart_recursive(ordered, unordered)
  return {
    'chart': found[0],
    'coords': found[1]
  } if (found := _get_chart_recursive([], set(coords))) and found[0] else None

chart_test = _get_chart({y, x, z}, EEE, debug=True)
