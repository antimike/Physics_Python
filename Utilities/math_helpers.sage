from functools import wraps
from collections import namedtuple
from sage.manifolds.operators import *
from sage.manifolds.catalog import Sphere
from scipy import special as fns
from sympy import factorial2

sys.path.append('/home/user/Documents/Python/Utilities')
import debugger as debg

def _catch_NameError(fn):
    """_catch_NameError.
    Helper function/decorator to provide a helpful hint on any NameErrors thrown by functions that rely on specific variables (e.g., E, Z_0, B, c, k, etc.)

  :param fn: Function to decorate
    """
    @wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except NameError as e:
            import sys
            raise NameError(str(e) + ' Perhaps try calling math_helpers.initialize_EM_variables()?').with_traceback(sys.exec_info()[2])
    return wrapper

def initialize_EM_variables(subs=None):
    """initialize_EM_variables.
    Initializes variables used in EM computations: q, Q, c, k, Z_0, R, V, t, omega, epsilon_o, mu_0.
  Initializes and returns 'substitute_exprs,' a dictionary of substitutions indexed by variable name.  If optional 'subs' argument is provided, new dictionary is appended to it.

  :param subs: (Optional) Dictionary of existing 'substitute_exprs.'
    """
    q = var('q')
    c,k = var('c k')
    Z_0 = var('Z_0')
    R,V,t = var('R V t')
    omega, epsilon_0, mu_0 = var('omega epsilon_0 mu_0')
    subs = subs if subs is not None else {}
    subs |= {
        omega: k*c,
        epsilon_0: 1/(Z_0*c),
        mu_0: Z_0/c
    }
    return subs

Fields = namedtuple('Fields', ['E', 'H'], defaults=[0, 0])
Multipole = namedtuple('Multipole', ['l', 'm', 'a_E', 'a_M', 'fields', 'angular_power'], defaults=[0, 0, 0, 0, Fields(), 0])

# Variable definitions

em_subs = initialize_EM_variables()

EEE.<r,th,ph> = EuclideanSpace(coordinates='spherical')
g = EEE.metric()
sph = EEE.default_chart()
cart.<x, y, z> = EEE.cartesian_coordinates()
frame_sph = EEE.default_frame() # e_r, e_th, e_ph
frame_sph.set_name('e_sph', latex_symbol=[r"\vu{r}", r"\vu*{\theta}", r"\vu*{\phi}"])
r_hat, th_hat, ph_hat = frame_sph
frame_cart = cart.frame()
frame_cart.set_name('e_cart', latex_symbol=[r"\vu{x}", r"\vu{y}", r"\vu{z}"])
x_hat, y_hat, z_hat = frame_cart

r_vec = EEE.vector_field((r, 0, 0))

def pt_sph(r=r, th=th, ph=ph):
    return EEE((r, th, ph), chart=sph)

def pt_cart(x=x, y=y, z=z):
    return EEE((x, y, z), chart=cart)

def hermitian_conjugate(arg):
    try:
        return conjugate_scalar(arg)
    except AttributeError:
        return hermitian_conjugate_vector(arg)

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

"""cart_bounds = {x: [1, 2], y: [-2, 2], z: 4}
test_scalar = EEE.scalar_field(x^2 + y^2 + z^2, chart=cart)
surface_integral_scalar(test_scalar, cart_bounds)"""
def integral_coord_region(scalar, bounds):
    """integral_coord_region.
    Computes the integral of a scalar function over a region defined by either arbitrary bounds on coordinates of a certain chart, or by constraints of the form 'coord == const'.

    :param scalar: Function to be integrated (scalar field)
    :param bounds: Dictionary of bounds.  Entries should either be of the form `{x: [a, b]}`, where `x` is a coordinate variable and `a` and `b` are the corresponding (possibly functional) bounds, or `{x: A}`, where `A` is a constant.
    """
    # manifold = scalar._manifold
    manifold = scalar.domain()
    c = _get_chart(bounds.keys(), manifold)
    vol_element = manifold.volume_form().comp(c['chart'].frame())[1, 2, 3]
    ret = (scalar*vol_element)(manifold.point(c['coords'], chart=c['chart']))
    for var, bound in bounds.items():
        if isinstance(bound, list) or isinstance(bound, tuple):
            ret = integral(ret, var, *bound)
        else:
            ret = ret.subs(var==bound)
    return ret

# EM functions (mostly from Jackson)

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
    return -i*r_vec.cross(scalar.gradient())

def X_lm_jackson(l, m):
    """X_lm_jackson.
    Vector spherical harmonic, defined by X_lm = [l(l+1)]^(-1/2)*LY_lm.
  See Jackson 9.119.

  :param l: Order of the multipole (angular momentum)
  :param m: Order of the multipole (magnetic)
    """
    return 1/sqrt(l*(l+1))*L_operator(Y_lm_jackson(l, m))

@_catch_NameError
def spherical_wavefront(l, outgoing, incoming, k=k):
    """spherical_wavefront.
    Returns a scalar field describing a spherical wavefront with specified outgoing and incoming coefficients.  Outgoing corresponds to h_l^1(kr), incoming to h_l^2(kr).
    See Jackson 9.113.

    :param l: Order of the multipole (angular momentum)
    :param outgoing: Coefficient of h_l^1(kr)
    :param incoming: Coefficient of h_l^2(kr)
    :param k: (Optional) wavevector of radiation.  If not provided, the variable 'k' is used.
    """
    return outgoing*spherical_hankel1(l, k*r) + incoming*spherical_hankel2(l, k*r)

@_catch_NameError
def a_lm_E_long_wavelength(l, m, Q_static, Q_induced=0):
    """a_lm_E_long_wavelength.
    Compute the multipole coefficients a_lm^E as a function of l, m, and the static and induced moments Q and Q_induced.
  See Jackson 9.169.

  :param l: Order of the multipole (angular momentum)
  :param m: Order of the multipole (magnetic)
  :param Q_static: Static multipole moment.  Should include the induced multipole moment due to magnetic induction.
  :param Q_induced: Electric multipole moment due to magnetic induction (default = 0)
    """
    return c*k^(l + 2)/(i*factorial2(2*l + 1))*sqrt((l + 1)/l)*(Q_static + Q_induced)

@_catch_NameError
def a_lm_M_long_wavelength(l, m, M_current, M_intrinsic):
    """a_lm_M_long_wavelength.
    Compute the multipole coefficients a_lm^E as a function of l, m, and the magnetic moments corresponding to currents (M_current) and intrinsic magnetization (M_intrinsic).
  See Jackson 9.171.

    :param l: Order of the multipole (angular momentum)
    :param m: Order of the multipole (magnetic)
    :param M_current: Magnetic multipole moment due to current
    :param M_intrinsic: Magnetic multipole moment due to intrinsic magnetization
    """
    return i*k^(l+2)/factorial2(2*l+1)*sqrt((l+1)/l)*(M_current + M_intrinsic)

@_catch_NameError
def Q_lm(l, m, charge_density, bounds):
    """Q_lm.
    Electric multipole moment due to a static charge distribution in the long-wavelength limit.
    See Jackson 9.170.

    :param l: Order (angular momentum)
    :param m: Order (magnetic)
    :param charge_density: Scalar function describing the charge distribution.  Can be confined to a surface or line if appropriate bounds are used.
    :param bounds: Bounds describing the extent of the charge distribution.  Can restrict the dimension in the same way that `integral_coord_region` allows.
    """
    return integral_coord_region(
        r^l*hermitian_conjugate(Y_lm_jackson(l, m))*charge_density, bounds)

@_catch_NameError
def Q_lm_induced(l, m, magnetization, bounds):
    """Q_lm_induced.
    Electric multipole moment induced by intrinsic magnetization in the long-wavelength limit.
    See Jackson 9.170.

    :param l: Order (angular momentum)
    :param m: Order (magnetic)
    :param magnetization: Intrinsic magnetization (vector field)
    :param bounds: Bounds describing the extent of the magnetization.  Can restrict the dimension in the same way that `integral_coord_region` allows.
    """
    return -i*k/(c*(l+1))*integral_coord_region(
        r^l*hermitian_conjugate(Y_lm_jackson(l, m))*div(r_vec.cross(magnetization)), bounds)

@_catch_NameError
def M_lm_current(l, m, currrent, bounds):
    """M_lm_current.
    Magnetic multipole moment caused by a currrent source in the long-wavelength limit.
    See Jackson 9.172.

    :param l: Order (angular momentum)
    :param m: Order (magnetic)
    :param currrent: Current density (vector field)
    :param bounds: Bounds describing the extent of the current source.  Can restrict the dimension in the same way that `integral_coord_region` allows.
    """
    return -1/(l+1)*integral_coord_region(
        r^l*hermitian_conjugate(Y_lm_jackson(l, m))*div(r_vec.cross(current)), bounds)

@_catch_NameError
def M_lm_intrinsic(l, m, magnetization, bounds):
    """M_lm_intrinsic.
    Magnetic multipole moment caused by intrinsic magnetization in the long-wavelength limit.
    See Jackson 9.172.

    :param l: Order (angular momentum)
    :param m: Order (magnetic)
    :param magnetization: Magnetization density (vector field)
    :param bounds: Bounds describing the extent of the magnetization.  Can restrict the dimension in the same way that `integral_coord_region` allows.
    """
    return -integral_coord_region(
        r^l*hermitian_conjugate(Y_lm_jackson(l, m))*div(magnetization), bounds)

@_catch_NameError
def E_lm_E_long_wavelength_expanded(l, m, a, k=k, Z_0=Z_0):
    """E_lm_E_long_wavelength.
    Electric field of an outgoing electric multipole with coefficient a.
    Expanded from the form given in Jackson 9.122.

  :param l: Order of the multipole (angular momentum)
  :param m: Order of the multipole (magnetic)
  :param a: Multipole coefficient
  :param k: (Optional) wavevector of radiation.  If not provided, the variable 'k' is used.
  :param Z_0: (Optional) wave impedance.  If not provided, the variable 'Z_0' is used.
    """
    return i*Z_0*a/k \
        *(((l + 1)/(2*l + 1)*k*r*spherical_hankel1(l+1, k*r) \
           - l/(2*l + 1)*k*r*spherical_hankel1(l-1, k*r) + i*spherical_hankel1(l, k*r)) \
          *grad(Y_lm_jackson(l, m)) \
          - i*l*(l+1)/r*spherical_hankel1(l, k*r)*Y_lm_jackson(l, m)*frame_sph[1])

@_catch_NameError
def H_lm_E_long_wavelength_expanded(l, m, a, E=None, k=k, Z_0=Z_0):
    """_H_lm_E_long_wavelength.
    Returns the magnetic multipole field due to an outgoing electric multipole of given order with given coefficient.
  If optional argument E is provided, computes the result by taking the curl of E.
    Expanded from the form given in Jackson 9.122.

  :param l: Order of the multipole (angular momentum)
  :param m: Order of the multipole (magnetic)
  :param a: Coefficient of the (electric) multipole
  :param E: (Optional) Electric field of the multipole
  :param k: (Optional) wavevector of radiation.  If not provided, the variable 'k' is used.
  :param Z_0: (Optional) wave impedance.  If not provided, the variable 'Z_0' is used.
    """
    if E is None:
        E = E_lm_E_long_wavelength(l, m, a, k=k, Z_0=Z_0)
    return -i/(k*Z_0)*curl(E)

@_catch_NameError
def multipole_fields_lm(l, m, A_E_outgoing, A_M_outgoing,
                        A_E_incoming=0, A_M_incoming=0, k=k, Z_0=Z_0):
    """multipole_fields_lm.
    Returns a 'Field' namedtuple with the fields of a pure (l, m) multipole with specified outgoing and incoming moments.
    See Jackson 9.122.

    :param l: Order of the multipole (angular momentum)
    :param m: Order of the multipole (magnetic)
    :param A_E_outgoing: Outgoing electric moment
    :param A_M_outgoing: Outgoing magnetic moment
    :param A_E_incoming: Incoming electric moment (default = 0)
    :param A_M_incoming: Incoming magnetic moment (default = 0)
    :param k: (Optional) wavevector of radiation.  If not provided, the variable 'k' is used.
    :param Z_0: (Optional) wave impedance.  If not provided, the variable 'Z_0' is used.
    """
    a_E_f_l = spherical_wavefront(l, A_E_outgoing, A_E_incoming)
    a_M_g_l = spherical_wavefront(l, A_M_outgoing, A_M_incoming)
    E_lm = Z_0*(i/k*curl(a_E_f_l*X_lm_jackson(l, m)) + a_M_g_l*X_lm_jackson(l, m))
    H_lm = a_E_f_l*X_lm_jackson(l, m) - i/k*curl(a_M_g_l*X_lm_jackson(l, m))
    return Fields(E=E_lm, H=H_lm)

@_catch_NameError
def multipole_power_cross_section_pure(l, m, a, k=k, Z_0=Z_0):
    """multipole_power_cross_section_pure.
    Returns the differential cross-section of a "pure" multipole.
  See Jackson 9.151.

    :param l: Order of the multipole (angular momentum)
    :param m: Order of the multipole (magnetic)
    :param a: Coefficient of the multipole
    :param k: (Optional) wavevector of radiation.  If not provided, the variable 'k' is used.
    :param Z_0: (Optional) wave impedance.  If not provided, the variable 'Z_0' is used.
    """
    return Z_0/(2*k^2)*norm(a + 0*i)*vector_complex_norm_squared(X_lm_jackson(l, m))

@_catch_NameError
def multipole_power_cross_section(multipoles, k=k, Z_0=Z_0):
    """multipole_power_cross_section.
    Computes the time-averaged power radiated per solid angle (cross-section) of a given collection of multipoles.
    See Jackson 9.150.

    :param multipoles: List or iterable of multipole namedtuples
    :param k: (Optional) wavevector of radiation.  If not provided, the variable 'k' is used.
    :param Z_0: (Optional) wave impedance.  If not provided, the variable 'Z_0' is used.
    """
    return Z_0/(2*k^2)*norm(sum(
        ((-i)^(l+1)*(_m.a_E*X_lm_jackson(_m.l, _m.m).cross(r_vec) + _m.a_M*X_lm_jackson(_m.l, _m.m)) for _m in multipoles)
    ) + 0*i)

@_catch_NameError
def multipole_power_total(multipoles, k=k, Z_0=Z_0):
    """multipole_power_total.
    Computes the time-averaged total power radiated by a given collection of multipoles.
    See Jackson 9.155.

    :param multipoles: List or iterable of multipole namedtuples
    :param k: (Optional) wavevector of radiation.  If not provided, the variable 'k' is used.
    :param Z_0: (Optional) wave impedance.  If not provided, the variable 'Z_0' is used.
    """
    return Z_0/(2*k^2)*sum(
        (norm(_m.a_E + 0*i) + norm(_m.a_M + 0*i) for _m in multipoles)
    )

def scalar_potential_azimuthal(A, B):
    """scalar_potential_azimuthal.
    Returns the scalar electric potential assuming azimuthal symmetry and given iterables of coefficients a_l and b_l.

  :param A: Iterable containing the coefficients a_l of r^l
  :param B: Iterable containing the coefficients b_l of r^(-(l + 1))
    """
    return EEE.scalar_field(sum(
        ((a*r^l + b/r^(l + 1))*legendre_P(l, cos(th)) for l, (a, b) in enumerate(zip(A, B)))
    ))


"""_get_chart([y, z, x], EEE, debug=True)"""
@debg.debug(alias='dbg',
            recursion_state='State: ordered = {}, unordered = {}, _unordered = {}',
            testing='Testing: ordered = {}, unordered = {}',
            vline='-'*25,
            testing_post='Testing {} from ordering {}', found='Found: {}', not_found='Not found!',
            passing='Received ordered = {}, unordered = {}; recursing further')
def _get_chart(coords, manifold, **kwargs):
    dbg = kwargs.get('dbg')
    def _get_chart_recursive(ordered, unordered):
        dbg.indent(len(ordered) + 1)
        _unordered, ret = set(unordered), None
        while ret is None and any(unordered):
            first_arg = ordered + [coord := unordered.pop()]
            second_arg = _unordered.difference({coord})
            dbg.vline().indent().recursion_state(ordered, unordered, _unordered).testing(first_arg, second_arg).vline()
            ret = _test(first_arg, second_arg)
            dbg.vline().indent().found(ret).vline()
        return ret
    def _test(ordered, unordered):
        if len(unordered) == 0:
            try:
                string = ' '.join(map(str, ordered)).strip()
                dbg.testing_post(string, ordered)
                return (manifold.get_chart(string), ordered)
            except KeyError:
                dbg.not_found()
                return None
        else:
            dbg.passing(ordered, unordered)
            return _get_chart_recursive(ordered, unordered)
    return {
      'chart': found[0],
      'coords': found[1]
    } if (found := _get_chart_recursive([], set(coords))) and found[0] else None
