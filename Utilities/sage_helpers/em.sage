
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

@_catch_NameError
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
@_catch_NameError
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

@_catch_NameError
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

@_catch_NameError
def E_wavefront_lm(l, m, E):
    """E_wavefront_lm.
    Returns the function a_E(l, m)*f_l(k*r), in the notation of Jackson's eq. 9.123.

    :param l: Order of the multipole (angular momentum)
    :param m: Order of the multipole (magnetic)
    :param E: Electric field

    >>> _l, _m = (1, 0)
    >>> _out, _in = (1, 0)
    >>> f_l = spherical_wavefront(_l, _out, _in)
    >>> H = f_l*L_operator(Y_lm_jackson(_l, _m))      # From Jackson 9.118 (pure-electric multipole)
    >>> E = i*Z_0/k*curl(H)
    >>> wavefront = E_wavefront_lm(_l, _m, E)
    >>> test_equality_radial(f_l, wavefront, r_min=1, r_max=10)
    >>> test_equality_general(f_l, wavefront, bounds={r: [2, 10], th: th, ph: ph})
    >>> wavefront.display(chart=sph)
    E^3 --> R
    (r, th, ph) |--> -1/3*(3*Z_0*k*r + 3*I*Z_0)*e^(I*k*r)/(Z_0*k^2*r^2)
    >>> f_l_confirm = -k/Z_0/(_l*(_l+1))/Y_lm_jackson(_l, _m)*r_vec.dot(E)      # See Jackson 9.117 (defn. of f_l in terms of r.E)
    >>> f_l_confirm.display(chart=sph)
    E^3 --> R
    (r, th, ph) |--> -(k*r + I)*e^(I*k*r)/(k^2*r^2)
    >>> (f_l_confirm - f_l)(pt_sph()).simplify()
    0
		>>> a_test = 1
		>>> H_test = a_test*spherical_wavefront(1, 1, 0)*X_lm_jackson(1, 0)
		>>> E_test = i*Z_0/k*curl(H_test)
		>>> wf_test = E_wavefront_lm(1, 0, E_test)
		>>> wf_test.display(chart=sph)
		>>> simplify_trig(wf_test/spherical_wavefront(1, 1, 0))
    1
    """
    # Note the division by r^2, since we're integrating against d\Omega
    # See implementation of `integral_coord_region`
    # TODO: Fix this in a better way
    # Also note the factor of l(l+1), as opposed to Jackson's sqrt(l(l+1))...possible typo in Jackson
    # TODO: Check this
    # DONE: Checked, Jackson's right :(
    return -k/Z_0/r^2/sqrt(l*(l+1))*EEE.scalar_field(
        integral_coord_region(
            hermitian_conjugate(Y_lm_jackson(l, m))*r_vec.dot(E),
            {r: r, th: [0, pi], ph: [0, 2*pi]}
        )
    )

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
        r^l*hermitian_conjugate(Y_lm_jackson(l, m))*charge_density, bounds
    )

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

def E_dipole_fields(moment):
    """E_dipole_fields.
    Returns the E and H fields of a dipole oriented along z_hat with the given moment

    :param moment: Electric dipole moment.  UNITS: Same as H
    """
    A = (-i*mu_0*k*c*e^(i*k*r)*moment/(4*pi*r)*z_hat)
    H = H_from_A_free_space(A)
    return Fields(E=E_from_A_free_space(A, H), H=H)

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
    return -i*Z_0*a/k \
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
        E = E_lm_E_long_wavelength_expanded(l, m, a, k=k, Z_0=Z_0)
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
    return Z_0/(2*k^2)*norm(a + 0*i)*complex_norm(X_lm_jackson(l, m))

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
