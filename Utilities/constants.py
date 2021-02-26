from pint import UnitRegistry

ureg = UnitRegistry()
_ = ureg

def unitless(num):
    return ureg.Quantity(num, '')

# Boltzmann's constant
# Units: m**2*kg*s**-2*K**-1
k_B_ = 1.3806e-23*ureg.meter**2*ureg.kilogram/(ureg.second**2*ureg.kelvin)

# Planck's constant
# Units: m**2*kg*s**-1
h_ = 6.626e-34*ureg.meter**2*ureg.kilogram/ureg.second

# Avagadro's number
N_A_ = 6.022e23/ureg.mole

# Gas constant
R_ = k_B_*N_A_

# Modified Planck's constant
h_bar_ = 1.055e-34*ureg.meter**2*ureg.kilogram/ureg.second

# Speed of light
# Units: m/s
c_ = 2.9979e8*ureg.meter/ureg.second

# Charge of electron
# Units: C
e_charge_ = 1.602e-19*ureg.coulomb
e_ = 1.602e-19*ureg.coulomb

# Mass of electron
# Units: kg
e_mass_ = 9.109e-31*ureg.kilogram
m_e_ = 9.109e-31*ureg.kilogram

# Mass of proton
# Units: kg
m_p_ = 1.6726e-27*ureg.kilogram

# Permittivity of free space
# Units: m**-3*kg**-1*s**4*A**2
epsilon_0_ = 8.854e-12*ureg.meter**-3*ureg.second**4*ureg.ampere**2/ureg.kilogram
e_0_ = 8.854e-12*ureg.meter**-3*ureg.second**4*ureg.ampere**2/ureg.kilogram

# Permeability of free space
# Units: m*kg*s**-2*A**-2
mu_0_ = 1.2566e-6*ureg.meter*ureg.kilogram/(ureg.second**2*ureg.ampere**2)
