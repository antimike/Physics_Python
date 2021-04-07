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
  chart = _get_chart(set(bounds.keys()), manifold)
  vol_element = manifold.volume_form().comp(chart.frame())[1, 2, 3]
  ret = (scalar*vol_element)(manifold.point(
  for var, bound in bounds.items():
    if isinstance(bound, list):
      ret = integral(ret, var, *bound)
    else:
      

  pass

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

def _get_chart(coords, manifold):
  def _get_chart_recursive(ordered, unordered):
    _unordered = set(unordered)
    chart = None
    while chart is None and unordered:
      coord = unordered.pop()
      chart = _test([*ordered, coord], _unordered.difference({coord}))
    return chart
  def _test(ordered, unordered):
    if len(unordered) == 1:
      string = ' '.join(map(str, ordered.append(unordered.pop()))).strip()
      try:
        return manifold.get_chart(string)
      except KeyError:
        return None
    else:
      return _get_chart_recursive(ordered, unordered)
  return _get_chart_recursive([], coords)

_get_chart({z, x, y}, EEE)
