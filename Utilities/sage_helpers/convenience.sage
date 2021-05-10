from .debug import debg
from deprecation import deprecated

""" Manifold helpers """
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

""" Infixes and supporting helper functions """
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

""" Decorators """
@deprecated(details="This was never a good idea.  Don't use it.")
def _catch_NameError(fn):
    """_catch_NameError.
    Helper function/decorator to provide a helpful hint on any NameErrors thrown by functions that rely on specific variables (e.g., E, Z_0, B, c, k, etc.)

  :param fn: Function to decorate
    """
    @sage_wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except NameError as e:
            import sys
            raise NameError(str(e) + ' Perhaps try calling math_helpers.initialize_EM_variables()?').with_traceback(sys.exec_info()[2])
    return wrapper
