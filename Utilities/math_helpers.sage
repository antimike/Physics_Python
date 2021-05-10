from functools import wraps, cache
from collections import namedtuple
from sage.manifolds.operators import *
from sage.manifolds.catalog import Sphere
from scipy import special as fns
from sympy import factorial2, assoc_legendre
from pint import UnitRegistry
from deprecation import deprecated
from typing import Optional
import logging

sys.path.append('/home/user/Documents/Python')
import Utilities.debugger as debg
from Utilities.sage_helpers.direction import Direction

""" Helpers and decorators """
@debg.debug(alias='dbg',
            recursion_state='State: ordered = {}, unordered = {}, _unordered = {}',
            testing='Testing: ordered = {}, unordered = {}',
            vline='-'*25,
            testing_post='Testing {} from ordering {}', found='Found: {}', not_found='Not found!',
            passing='Received ordered = {}, unordered = {}; recursing further')
def _get_chart(coords, manifold, **kwargs):
    """_get_chart.

    :param coords: String containing the coordinates of the chart to be found, e.g., 'y x z' (order doesn't matter)
    :param manifold: Manifold on which the chart is defined
    :param kwargs: Keywords consumed and defined by @debug.debug decorator

    >>> _get_chart([y, z, x], EEE, dbg=False)
    >>> {'chart': Chart (E^3, (x, y, z)), 'coords': [x, y, z]}
    """
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

def _modify_scalar_field(scalar, transformation, **kwargs):
    """_modify_scalar_field.
    Transforms the coordinate expression of a scalar field

    :param scalar: Scalar field to transform
    :param transformation: Callable to apply to scalar field
    :kwargs: Keywords received from public caller and passed to scalar's `set_name` method

    >>> sf = EEE.scalar_field(x^2 - y + i*z^3)
    >>> _modify_scalar_field(sf, conjugate)
    >>> sf.expr()
    -I*z^3 + x^2 - y
    """
    try:
        scalar.set_expr(transformation(scalar.expr()))
        scalar.set_name(**kwargs)
    # Toss AttributeErrors back to caller to support duck typing
    except AttributeError:
        raise
    # Handle other exceptions by looping over manifold charts one by one
    except:
        logging.warning("""
    Transformation of scalar '{field}' using generic call scalar.expr() failed
    Attempting a loop through manifold coordinate charts
        """.format(field=scalar))
        success = False
        for chart in scalar._manifold.atlas():
            try:
                scalar.set_expr(transformation(scalar.coord_function(chart=chart)), chart=chart)
                scalar.set_name(**kwargs)
                success = True
                break
            except:
                logging.warning("""
    Cannot set transform expression for scalar field '{field}' in chart '{chart}' of manifold '{man}'
                """.format(field=scalar, chart=chart, man=scalar._manifold))
        # Something went horribly wrong if we reach the following block
        if not success:
            logging.error("""
    Cannot transform coordinate expression for scalar field '{field}' in any chart of manifold '{man}'
            """.format(field=scalar, man=scalar._manifold))
            raise

def _modify_vector_field(vector, transformation, **kwargs):
    """_modify_vector_field.
    Transforms the coordinate expressions of a vector field one by one

    :param vector: Vector field to transform
    :param transformation: Callable to apply to vector field components

    >>> vf = EEE.vector_field(-x, y+z^2 - i*sin(x), i*tan(y), name="Test", frame=frame_cart)
    >>> _modify_vector_field(vf, conjugate)
    >>> vf.display_comp(frame=frame_cart, chart=cart)
    Test^x = -x
    Test^y = z^2 + y + I*sin(x)
    Test^z = -I*sin(y)/cos(y)
    """
    try:
        vector.set_comp()[:] = [transformation(comp.expr()) for comp in vector.comp()[:]]
    # Toss AttributeErrors back to caller to support duck typing
    except AttributeError:
        raise
    # Handle other exceptions by looping over manifold charts one by one
    except:
        logging.warning("""
    Transformation of vector field '{field}' using generic call vector.set_comp() failed
    Attempting a loop through manifold vector frames
        """.format(field=vector))
        success = False
        for frame in vector._domain.frames():
            try:
                vector.set_comp(basis=frame)[:] = \
                    [transformation(comp.expr()) for comp in vector.comp(basis=frame)[:]]
                break
            except:
                logging.warning("""
    Cannot set transform component expressions for vector field '{field}' in basis '{basis}' of manifold '{man}'
                """.format(field=vector, basis=frame, man=vector._domain))
        # Something went horribly wrong if we reach the following block
        if not success:
            logging.error("""
    Cannot transform coordinate expression for vector field '{field}' in any basis of manifold '{man}'
            """.format(field=vector, man=vector._domain))
            raise
    # Set 'name' and 'latex_name' properties of arg
    if kwargs.get('name') is not None or kwargs.get('latex_name') is not None:
        vector.set_name(**kwargs)

def _apply_to_field(fn, field, **kwargs):
    """_apply_to_field.
    Apply a function to the components and/or chart expressions of a scalar or vector field and return the result as a new object.

    :param fn: Function (callable) to apply
    :param field: Field to transform
    :kwarg name: Name to pass to the new object
    :kwargs: Keywords received from public caller and passed to _modify_scalar_field or _modify_vector_field
    """
    new = field.copy()
    try:
        _modify_scalar_field(new, fn, **kwargs)
    except:
        _modify_vector_field(new, fn, **kwargs)
    return new

def _get_new_obj_names(obj, **kwargs):
    fname = lambda fn, on: fn and r"{fname}({objame})".format(fname=fn, objname=on)
    new_oname = kwargs.get('new_name') or fname(kwargs.get('func_name'), obj._name)
    new_lname = kwargs.get('new_latex_name') or fname(kwargs.get('func_latex_name'), obj._latex_name)
    return {'name': new_oname, 'latex_name': new_lname}

def _apply_to(fn, obj, **kwargs):
    """_apply_to.
    Returns a new expression, scalar field, or vector field representing the result of calling `fn` on the components and/or coordinate expressions of `field`.

    :param fn: Function to apply (componentwise in the case of vectors)
    :param obj: Object (expression or field) to apply `fn` to
    :kwargs: Keywords received from public caller and passed to `_get_new_obj_names`
    """
    if hasattr(obj, '_domain'):
        return _apply_to_field(fn, obj, **_get_new_obj_names(obj, **kwargs))
    else:
        return fn(obj)

@infix_operator('or')
def apply_to(fn, obj, new_name=None, new_latex_name=None, func_name=None, func_latex_name=None):
    """apply_to.
    Applies a function to the basis and/or chart expressions of a obj and returns the result as a new obj.
    Does not modify the original obj.

    :param fn: Function to apply
    :param obj: Object to apply `fn` to
    :kwarg new_name: Name to pass to the new object.  Default = None
    :kwarg new_latex_name: Latex name to pass to the new object.  Default = None
    :kwarg func_name: Name of function to apply.  Used to construct `name` if `name` is not provided.  Default = None
    :kwarg func_latex_name: Latex name of function to apply.  Used to construct `latex_name` if `latex_name` is not provided.  Default = None

    >>> sf = EEE.scalar_field(x^2 - 2*x + 4*i*z)
    >>> sf_conj = conjugate |apply_to| sf
    >>> sf_conj.display(chart=cart)
    E^3 --> R
    (x, y, z) |--> x^2 - 2*x - 4*I*z
    >>> vf = EEE.vector_field(sin(x) + i*cos(x), x^2 - z^3, i*y + tan(x), frame=frame_cart)
    >>> vf_conj = conjugate |apply_to| vf
    >>> vf_conj.display_comp(frame=frame_cart, chart=cart)
    X^x = -I*cos(x) + sin(x)
    X^y = -z^3 + x^2
    X^z = (-I*y*cos(x) + sin(x))/cos(x)
    """
    if not callable(fn):
        logging.error("Tried to apply non-callable '{func}' to object '{obj}'".format(
            func=fn, obj=obj))
        raise ValueError("Parameter '{func}' must be callable".format(func=fn))
    kwargs = {'new_name': new_name, 'new_latex_name': new_latex_name,
              'func_name': func_name, 'func_latex_name': func_latex_name}
    return _apply_to(fn, obj, **kwargs)

@infix_operator('or')
def modify_with(obj, fn, new_name=None, new_latex_name=None, func_name=None, func_latex_name=None):
    """modify_with.
    Modifies the passed object in-place by applying the given function

    :param fn: Function to apply
    :param obj: Object to apply `fn` to.  Must expose `is_mutable()` method
    :kwarg new_name: New name to give object.  Default = None
    :kwarg new_latex_name: New latex_name to give object.  Default = None
    :kwarg func_name: Name of function to apply.  Used to construct `name` if `name` is not provided.  Default = None
    :kwarg func_latex_name: Latex name of function to apply.  Used to construct `latex_name` if `latex_name` is not provided. Default = None

    >>> sf = EEE.scalar_field(x^2 - 2*x + 4*i*z, name='Test')
    >>> sf |modify_with| conjugate
    >>> sf.display(chart=cart)
    E^3 --> R
    (x, y, z) |--> x^2 - 2*x - 4*I*z
    >>> vf = EEE.vector_field(sin(x) + i*cos(x), x^2 - z^3, i*y + tan(x), frame=frame_cart)
    >>> vf |modify_with| conjugate
    >>> vf.display_comp(frame=frame_cart, chart=cart)
    X^x = -I*cos(x) + sin(x)
    X^y = -z^3 + x^2
    X^z = (-I*y*cos(x) + sin(x))/cos(x)
    >>> (x^2 - 2*x + 3) |modify_with| conjugate
    AttributeError                            Traceback (most recent call last)
    ...
    AttributeError: 'sage.symbolic.expression.Expression' object has no attribute 'is_mutable'
    """
    if not callable(fn):
        logging.error("Tried to apply non-callable '{func}' to object '{obj}'".format(
            func=fn, obj=obj))
        raise ValueError("Parameter '{func}' must be callable".format(func=fn))
    if obj.is_mutable():
        kwargs = {'new_name': new_name, 'new_latex_name': new_latex_name,
                  'func_name': func_name, 'func_latex_name': func_latex_name}
        try:
            _modify_vector_field(obj, fn, **_get_new_obj_names(obj, **kwargs))
        except:
            _modify_scalar_field(obj, fn, **_get_new_obj_names(obj, **kwargs))
    else:
        logging.error("Tried to modify immutable object '{obj}' with function '{func}'".format(
            obj=obj, func=fn))
        raise ValueError("This function can only be called with mutable arguments!")

""" Variable definitions """
def initialize_EM_variables(subs=None):
    """initialize_EM_variables.
    Initializes variables used in EM computations: q, Q, c, k, Z_0, R, V, t, omega, epsilon_o, mu_0.
    Initializes and returns 'substitute_exprs,' a dictionary of substitutions indexed by variable name.
    If optional 'subs' argument is provided, new dictionary is appended to it.

    :param subs: (Optional) Dictionary of existing 'substitute_exprs.'
    """
    # Complex variables
    k, omega = var('k omega')
    # Real variables
    c, q, Z_0, R, t, epsilon_0, mu_0 = var('c q Z_0 R t epsilon_0 mu_0', domain='real')
    subs = subs if subs is not None else {}
    subs |= {
        omega: k*c,
        epsilon_0: 1/(Z_0*c),
        mu_0: Z_0/c
    }
    return subs

Fields = namedtuple('Fields', ['E', 'H'], defaults=[0, 0])
Multipole = namedtuple('Multipole', ['l', 'm', 'a_E', 'a_M', 'fields', 'angular_power'], defaults=[0, 0, 0, 0, Fields(), 0])

em_subs = initialize_EM_variables()
ureg = UnitRegistry()
Q_ = ureg.Quantity

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

r_vec = r*r_hat

def pt_sph(r=r, th=th, ph=ph):
    return EEE((r, th, ph), chart=sph)

def pt_cart(x=x, y=y, z=z):
    return EEE((x, y, z), chart=cart)

""" Pure math functions """
def trig_to_exp(expr):
    """trig_to_exp.
    Convenience wrapper around Maxima's `exponentialize` function

    :param expr: Expression to exponentialize
    """
    return sageobj(expr._maxima_().exponentialize())

def exp_to_trig(expr):
    """exp_to_trig.
    Convenience wrapper around Maxima's `demoivre` function

    :param expr: Expression to de-exponentialize
    """
    return sageobj(expr._maxima_().demoivre())

# TODO: Test `trig_reduce`
def simplify_trig(fn):
    """simplify_trig.
    Convenience wrapper around `exp_to_trig`

    :param fn: Function or expression to simplify
    """
    try:
        return exp_to_trig(fn(pt_sph())).trig_reduce()
    except:
        return exp_to_trig(fn)

@deprecated(details="Use 'apply_to' with callable 'conjugate' instead")
def hermitian_conjugate(arg):
    """hermitian_conjugate.
    A single-dispatch convenience wrapper around `_hermitian_conjugate_vector` and `conjugate_scalar`

    :param arg: Field to be conjugated (scalar or vector)
    """
    try:
        return _conjugate_scalar(arg)
    except AttributeError:
        return _hermitian_conjugate_vector(arg)

def complex_norm(arg):
    """complex_norm.
    Single-dispatch convenience wrapper around `_scalar_complex_norm` and `_vector_complex_norm`

    :param arg: Field (scalar or vector) whose complex norm should be returned
    """
    try:
        return _scalar_complex_norm(arg)
    except AttributeError:
        return _vector_complex_norm(arg)

def complex_magnitude(arg):
    """complex_magnitude.
    Convenience wrapper around sqrt of `complex_norm`

    :param arg: Expression, scalar, or vector field to find complex magnitude of
    """
    return sqrt(complex_norm(arg))

@deprecated(details="Use 'apply_to' with callable 'conjugate' instead")
def _conjugate_scalar_field(scalar, manifold):
    """_conjugate_scalar_field.
    Helper function to handle the scalar field case of the single-dispatch function `conjugate_scalar`

    :param scalar: Field defined on `manifold` to be conjugated
    :param manifold: Base manifold
    """
    ret = manifold.scalar_field()
    for chart in manifold.atlas():
        try:
            ret.add_expr(conjugate(scalar.coord_function(chart=chart).expr() + 0*i), chart=chart)
        except:
            logging.warn("Cannot add conjugate expression for scalar field in chart %s of manifold %s" \
                         % (chart, manifold))
    return ret

@deprecated(details="Use 'apply_to' with callable 'conjugate' instead")
def _conjugate_scalar(scalar):
    """conjugate_scalar.
    If input is a scalar field, returns the complex conjugate as another scalar field on the same manifold.
    If input is an expression, returns the complex conjugate as another expression.

    :param scalar: Scalar field or expression to conjugate
    """
    try:
        manifold = scalar._manifold
        return _conjugate_scalar_field(scalar, manifold)
    except AttributeError:
        return conjugate(scalar + 0*i)

def _scalar_complex_norm(scalar):
    """_scalar_complex_norm.
    If input is a scalar field, returns the norm-squared as another scalar field on the same manifold
    If input is an expression, returns the norm-squared as another expression

    :param scalar: The scalar field to conjugate
    """
    return apply_to(conjugate, scalar)*scalar

@deprecated(details="Use 'apply_to' with callable 'conjugate' instead")
def _hermitian_conjugate_vector(vector):
    """_hermitian_conjugate_vector.
    Wraps the Hermitian conjugate of a vector field as a differential 1-form on the same manifold.

  :param vector: Vector field to conjugate
    """
    manifold = vector._domain
    ret = manifold.diff_form(1)
    for chart in manifold.atlas():
        for j in range(1, manifold.dim() + 1):
            ret.add_comp(chart.frame())[j] = conjugate(vector.comp(chart.frame())[j].expr() + 0*i)
    return ret

def _vector_complex_norm(v):
    """_vector_complex_norm.
    Returns the norm-squared of a vector field with complex components, as a scalar field on the same manifold.

    :param v: Vector field to find norm of
    """
    return apply_to(conjugate, v)['_i']*v['i']

def integral_coord_region(scalar, bounds):
    """integral_coord_region.
    Computes the integral of a scalar function over a region defined by either arbitrary bounds on coordinates of a certain chart, or by constraints of the form 'coord == const'.

    :param scalar: Function to be integrated (scalar field)
    :param bounds: Dictionary of bounds.  Entries should either be of the form `{x: [a, b]}`, where `x` is a coordinate variable and `a` and `b` are the corresponding (possibly functional) bounds, or `{x: A}`, where `A` is a constant.

    >>> cart_bounds = {x: [1, 2], y: [-2, 2], z: 4}
    >>> test_scalar = EEE.scalar_field(x^2 + y^2 + z^2, chart=cart)
    >>> integral_coord_region(test_scalar, cart_bounds)
    236/3
    >>> integral_coord_region(hermitian_conjugate(X_lm_jackson(1, 0))['_i'] \
    ...:    * X_lm_jackson(3, 2)['i'], {r: r, th: [0, pi], ph: [0, 2*pi]})
    0
    >>> integral_coord_region(hermitian_conjugate(X_lm_jackson(4, 2))['_i'] \
    ...:    * (r_vec.cross(X_lm_jackson(3, -1)))['i'],
    ...:    {r: r, th: [0, pi], ph: [0, 2*pi]})
    0
    >>> integral_coord_region(hermitian_conjugate(Y_lm_jackson(2, 1))*Y_lm_jackson(2, 1),
    ...:    {r: r, th: [0, pi], ph: [0, 2*pi]})
    r^2
    """
    manifold = scalar.domain()
    c = _get_chart(bounds.keys(), manifold)
    vol_element = manifold.volume_form().comp(c['chart'].frame())[1, 2, 3]
    ret = (scalar*vol_element)(manifold.point(c['coords'], chart=c['chart']))
    for var, bound in bounds.items():
        if isinstance(bound, list) or isinstance(bound, tuple):
            ret = integral(ret, var, *bound)
        else:
            try:
                ret = ret.subs(var==bound)
            except AttributeError:              # 'int' object has no attribute 'subs'
                pass
    return ret

""" EM functions (mostly from Jackson) """
""" Spherical harmonics """
def Y_lm_test(l=None, m=None):
    return EEE.scalar_field(
        sqrt((2*l + 1)*factorial(l - m)/(4*pi*factorial(l + m)))
        * assoc_legendre(l, m, cos(th))*e^(i*m*ph)
    )

def Y_lm_jackson(l, m):
    """Y_lm_jackson.
    Spherical harmonic, with phase and normalization convention as in Jackson.
  See Jackson 3.53.

  :param l: Order (angular momentum)
  :param m: Order (magnetic)
    """
    if l < 0 or abs(m) > l:
        raise ValueError("The parameters (m, l) must satisfy 0 <= l and abs(m) <= abs(l)")
    #return EEE.scalar_field(spherical_harmonic(l, m, th, ph)*(-1)^m)
    return Y_lm_test(l, m)

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

def spherical_wavefront(l, outgoing, incoming, k=k):
    """spherical_wavefront.
    Returns a scalar field describing a spherical wavefront with specified outgoing and incoming coefficients.  Outgoing corresponds to h_l^1(kr), incoming to h_l^2(kr).
    See Jackson 9.113.

    :param l: Order of the multipole (angular momentum)
    :param outgoing: Coefficient of h_l^1(kr)
    :param incoming: Coefficient of h_l^2(kr)
    :param k: (Optional) wavevector of radiation.  If not provided, the variable 'k' is used.

    >>> _l, _out, _in = (2, 1, 0)
    >>> wavefront = spherical_wavefront(_l, _out, _in)
    >>> wavefront.display(chart=sph)
    E^3 --> R
    (r, th, ph) |--> spherical_hankel1(2, k*r)
    >>> wavefront(pt_sph())
    """
    return EEE.scalar_field(outgoing*spherical_hankel1(l, k*r) + incoming*spherical_hankel2(l, k*r))

""" Multipole moments """
def a_lm_E_long_wavelength(l, m, Q_static, Q_induced=0):
    """a_lm_E_long_wavelength.
    Compute the multipole coefficients a_lm^E as a function of l, m, and the static and induced moments Q and Q_induced.
    Valid in the limit kr << 1.
    See Jackson 9.169.

    :param l: Order of the multipole (angular momentum)
    :param m: Order of the multipole (magnetic)
    :param Q_static: Static multipole moment.  Should include the induced multipole moment due to magnetic induction.
    :param Q_induced: Electric multipole moment due to magnetic induction (default = 0)

    >>> a_lm_E_long_wavelength(2, 1, 1)
    -1/15*I*sqrt(3/2)*c*k^4
    """
    return c*k^(l + 2)/(i*factorial2(2*l + 1))*sqrt((l + 1)/l)*(Q_static + Q_induced)

def a_lm_M_long_wavelength(l, m, M_current, M_intrinsic):
    """a_lm_M_long_wavelength.
    Compute the multipole coefficients a_lm^E as a function of l, m, and the magnetic moments corresponding to currents (M_current) and intrinsic magnetization (M_intrinsic).
    Valid in the limit kr << 1.
    See Jackson 9.171.

    :param l: Order of the multipole (angular momentum)
    :param m: Order of the multipole (magnetic)
    :param M_current: Magnetic multipole moment due to current
    :param M_intrinsic: Magnetic multipole moment due to intrinsic magnetization
    """
    return i*k^(l+2)/factorial2(2*l+1)*sqrt((l+1)/l)*(M_current + M_intrinsic)

def E_wavefront_lm(l, m, E):
    """E_wavefront_lm.
    Returns the function a_E(l, m)*f_l(k*r), in the notation of Jackson's eq. 9.123.

    :param l: Order of the multipole (angular momentum)
    :param m: Order of the multipole (magnetic)
    :param E: Electric field

    >>> _l, _m, _out, _in = (1, 0, 1, 0)
    >>> f_l = spherical_wavefront(_l, _out, _in)
    >>> H = f_l*X_lm_jackson(_l, _m)
    >>> E = i*Z_0/k*curl(H)
    >>> wavefront = E_wavefront_lm(_l, _m, E)
    >>> test_equality_radial(f_l, wavefront, r_min=1, r_max=10)
    True
    >>> test_equality_general(f_l, wavefront, bounds={r: [2, 10], th: th, ph: ph})
    True
    >>> wavefront.display(chart=sph)
    E^3 --> R
    (r, th, ph) |--> -1/6*sqrt(2)*(3*sqrt(2)*Z_0*k*r + 3*I*sqrt(2)*Z_0)*e^(I*k*r)/(Z_0*k^2*r^2)
    """
    # Note the division by r^2, since we're integrating against d\Omega
    # See implementation of `integral_coord_region`
    # TODO: Fix this in a better way
    # Also note the factor of l(l+1), as opposed to Jackson's sqrt(l(l+1))...possible typo in Jackson
    # TODO: Check this
    # DONE: Checked, Jackson's right :(
    return -k/Z_0/r^2/sqrt(l*(l+1))*EEE.scalar_field(
        integral_coord_region(
            apply_to(conjugate, Y_lm_jackson(l, m))*r_vec.dot(E),
            {r: r, th: [0, pi], ph: [0, 2*pi]}
        )
    )

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
        r^l*apply_to(conjugate, Y_lm_jackson(l, m))*charge_density, bounds
    )

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
        r^l*apply_to(conjugate, Y_lm_jackson(l, m))*div(r_vec.cross(magnetization)), bounds)

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
        r^l*apply_to(conjugate, Y_lm_jackson(l, m))*div(r_vec.cross(current)), bounds)

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
        r^l*apply_to(conjugate, Y_lm_jackson(l, m))*div(magnetization), bounds)

""" Fields """
def H_from_A_free_space(A):
    """H_from_A_free_space.
    Returns the magnetic field H of a vector potential A in free space

    :param A: Vector potential
    """
    return curl(A)/mu_0

def E_from_A_free_space(A, H=None):
    """E_from_A_free_space.
    Computes E from a vector potential A in free space

    :param A: Vector potential
    :param H (optional): Magnetic field, to be provided if already computed
    """
    if H is None:
        H = H_from_A_free_space(A)
    return i*Z_0/k*curl(H)

def E_dipole_fields(moment, direction: Optional[Direction] = None) -> Fields:
    """E_dipole_fields.
    Returns the E and H fields of a dipole oriented along z_hat with the given moment

    :param moment: Electric dipole moment.  UNITS: Same as H

    >>> 
    """
    A = (-i*mu_0*k*c*e^(i*k*r)*moment/(4*pi*r)*z_hat)
    H = H_from_A_free_space(A)
    return Fields(E=E_from_A_free_space(A, H), H=H)

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
    return -i*Z_0*a/k \
        *(((l + 1)/(2*l + 1)*k*r*spherical_hankel1(l+1, k*r) \
           - l/(2*l + 1)*k*r*spherical_hankel1(l-1, k*r) + i*spherical_hankel1(l, k*r)) \
          *grad(Y_lm_jackson(l, m)) \
          - i*l*(l+1)/r*spherical_hankel1(l, k*r)*Y_lm_jackson(l, m)*frame_sph[1])

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

    >>> l=1: 'fields' off by factor of sqrt(2); 'exp_fields' off by overall minus sign
    >>> A_electric_dipole = -i*mu_0*omega/(4*pi)*e^(i*k*r)/r*z_hat          # Potential of a dipole with unit moment
    >>> def get_outgoing_wavefront_from_field(l, m, E):
    ...:    return -k/Z_0/sqrt(l*l+1) * integral_coord_region(_conjugate_scalar(Y_lm_jackson(l, m))*r*r_hat.dot(E),
    ...:        {th: [0, pi], ph: [0, 2*pi], r: 1})
    >>> @cache
    ...: def run_test_case(_l, _m, _A_E_out, _A_M_out):
    ...:    fields = multipole_fields_lm(_l, _m, _A_E_out, _A_M_out)
    ...:    exp_fields = Fields(
    ...:        E=E_lm_E_long_wavelength_expanded(_l, _m, _A_E_out),
    ...:        H=H_lm_E_long_wavelength_expanded(_l, _m, _A_E_out)
    ...:    )
    ...:    return {
    ...:        'q_numbers': (_l, _m),
    ...:        'fields': fields,
    ...:        'exp_fields': exp_fields,
    ...:        'wavefronts': {
    ...:            'fields': E_wavefront_lm(_l, _m, fields.E),
    ...:            'exp_fields': E_wavefront_lm(_l, _m, exp_fields.E),
    ...:            'expected': spherical_wavefront(_l, _A_E_out, 0)
    ...:        }
    ...:    }
    >>> test_cases = [[1, 0, 1, 0], [2, 0, 1, 0], [1, 1, 1, 0], [1, 2, 1, 0]]
    >>> results = (run_test_case(*L) for L in test_cases)
    >>> res = next(results)
    >>> test_equality_radial(res['wavefronts']['expected'], res['wavefronts']['fields'])
    >>> l, m, A_E_out, A_M_out = (2, 1, 5, 4)
    >>> (fields.E - E_exp).dot(r_hat).display()
    >>> r_hat.dot(fields.E).display()
    >>> r_hat.dot(fields.H)(pt_sph(r, th, ph))
    >>> test = -k/Z_0/sqrt(l*(l+1)) * integral_coord_region(_conjugate_scalar(Y_lm_jackson(l, m))*r*r_hat.dot(fields.E),
    ...:    {th: [0, pi], ph: [0, 2*pi], r: 1})
    >>> expect = spherical_wavefront(l, A_E_out, 0).subs(r=1)
    >>> (test - expect).subs(k=1, Z_0=1)
    >>> get_outgoing_wavefront_from_field(l, m, fields.E)
    >>> spherical_wavefront(l, A_E_out, 0).subs(r=1)
    >>> get_outgoing_wavefront_from_field(l, m, E_exp)
    >>> l_, m_, A_E_, A_M_ = (2, 0, 5, 0)
    >>> fields_ = multipole_fields_lm(l_, m_, A_E_, A_M_)
    >>> test_ = get_outgoing_wavefront_from_field(l_, m_, fields_.E).subs(Z_0=1, k=1)
    >>> expect_ = spherical_wavefront(l_, A_E_, 0).subs(k=1, r=1).simplify()
    >>> (test_ - expect_).expand()      # This should vanish
    """
    a_E_f_l = spherical_wavefront(l, A_E_outgoing, A_E_incoming)
    a_M_g_l = spherical_wavefront(l, A_M_outgoing, A_M_incoming)
    E_lm = Z_0*(i/k*curl(a_E_f_l*X_lm_jackson(l, m)) + a_M_g_l*X_lm_jackson(l, m))
    H_lm = a_E_f_l*X_lm_jackson(l, m) - i/k*curl(a_M_g_l*X_lm_jackson(l, m))
    return Fields(E=E_lm, H=H_lm)

""" Power """
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
    return Z_0/(2*k^2)*norm(a + 0*i)*complex_norm(X_lm_jackson(l, m))

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

""" Scalar potential """
def scalar_potential_azimuthal(A, B):
    """scalar_potential_azimuthal.
    Returns the scalar electric potential assuming azimuthal symmetry and given iterables of coefficients a_l and b_l.

  :param A: Iterable containing the coefficients a_l of r^l
  :param B: Iterable containing the coefficients b_l of r^(-(l + 1))
    """
    return EEE.scalar_field(sum(
        ((a*r^l + b/r^(l + 1))*legendre_P(l, cos(th)) for l, (a, b) in enumerate(zip(A, B)))
    ))

""" Testing and debugging """
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
    exprs = [complex_norm(qty1 - qty2).expr(), 1 - complex_norm(qty1/qty2).expr()]
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

""" Units """
# TODO: Implement some helpers / decorators here?
