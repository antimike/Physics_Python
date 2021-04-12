sys.path.append('/home/user/Documents/Python/Utilities')
import debugger as debg
#import math_helpers as mh
from math_helpers import *

cart_bounds = {x: [1, 2], y: [-2, 2], z: 4}
test_scalar = EEE.scalar_field(x^2 + y^2 + z^2, chart=cart)

test_vector = q/(4*pi*epsilon_0*r^2)*r_hat
integral_coord_region(test_vector.dot(r_hat), {r: R, ph: [0, 2*pi], th: [0, pi]})
integral_coord_region(test_vector.div(), {r: [0, R], ph: [0, 2*pi], th: [0, pi]})
integral_coord_region(test_vector, {r: R, ph: [0, 2*pi], th: [0, pi]})
EEE.scalar_field(1/r).laplacian()(pt_cart(0, 0, 0))

surface_integral_scalar(test_scalar, cart_bounds)


_get_chart([y, z, x], EEE, debug=True)

