    #>>> cart_bounds = {x: [1, 2], y: [-2, 2], z: 4}
    #>>> test_scalar = EEE.scalar_field(x^2 + y^2 + z^2, chart=cart)
    #>>> integral_coord_region(test_scalar, cart_bounds)
    #236/3
    #>>> integral_coord_region(hermitian_conjugate(X_lm_jackson(1, 0))['_i'] \
    #...:    * X_lm_jackson(3, 2)['i'], {r: r, th: [0, pi], ph: [0, 2*pi]})
    #0
    #>>> integral_coord_region(hermitian_conjugate(X_lm_jackson(4, 2))['_i'] \
    #...:    * (r_vec.cross(X_lm_jackson(3, -1)))['i'],
    #...:    {r: r, th: [0, pi], ph: [0, 2*pi]})
    #0
    #>>> integral_coord_region(hermitian_conjugate(Y_lm_jackson(2, 1))*Y_lm_jackson(2, 1),
    #...:    {r: r, th: [0, pi], ph: [0, 2*pi]})
    #r^2
def scalar_field(expr):
    M.<r,th,ph> = EuclideanSpace(coordinates='spherical')
    sph = M.default_chart()
    cart.<x, y, z> = M.cartesian_coordinates()
    return M.scalar_field(expr)

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

g = EEE.metric()
frame_sph = EEE.default_frame() # e_r, e_th, e_ph
frame_sph.set_name('e_sph', latex_symbol=[r"\vu{r}", r"\vu*{\theta}", r"\vu*{\phi}"])
r_hat, th_hat, ph_hat = frame_sph
frame_cart = cart.frame()
frame_cart.set_name('e_cart', latex_symbol=[r"\vu{x}", r"\vu{y}", r"\vu{z}"])
x_hat, y_hat, z_hat = frame_cart

r_vec = r*r_hat

