from deprecation import deprecated
import logging

logger = logging.getLogger()

""" Integrals """
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

""" Trig """
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

""" Conjugation"""
# TODO: Remove all deprecated functions and replace calls with updated ones
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
            logger.warn("Cannot add conjugate expression for scalar field in chart %s of manifold %s" \
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
    return _conjugate_scalar(scalar)*scalar

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
    return _hermitian_conjugate_vector(v)['_i']*v['i']
