from pint import UnitRegistry
from matplotlib import rc
import mendeleev as chem

rc('text', usetex=True)
rc('text.latex', preamble=r"\usepackage{siunitx}")

ureg = UnitRegistry()
def unitless(num):
    return ureg.Quantity(num, '')

# Boltzmann's constant
# Units: m**2*kg*s**-2*K**-1
k = 1.3806e-23*ureg.meter**2*ureg.kilogram/(ureg.second**2*ureg.kelvin)

# Planck's constant
# Units: m**2*kg*s**-1
h = 6.626e-34*ureg.meter**2*ureg.kilogram/ureg.second

# Avagadro's number
N_A = 6.022e23/ureg.mole

# Gas constant
R = k*N_A

# Modified Planck's constant
hbar = 1.055e-34*ureg.meter**2*ureg.kilogram/ureg.second

# Speed of light
# Units: m/s
c = 2.9979e8*ureg.meter/ureg.second

# Charge of electron
# Units: C
e_charge = 1.602e-19*ureg.coulomb

# Mass of electron
# Units: kg
e_mass = 9.109e-31*ureg.kilogram

# Mass of proton
# Units: kg
m_p = 1.6726e-27*ureg.kilogram

# Permittivity of free space
# Units: m**-3*kg**-1*s**4*A**2
epsilon_0 = 8.854e-12*ureg.meter**-3*ureg.second**4*ureg.ampere**2/ureg.kilogram

# Permeability of free space
# Units: m*kg*s**-2*A**-2
mu_0 = 1.2566e-6*ureg.meter*ureg.kilogram/(ureg.second**2*ureg.ampere**2)

def tabularize_columns(*cols):
    stringed_arrs = map(lambda arr: map(lambda x: str(x), arr), cols)
    return r" \\ ".join([r" & ".join(tuple) for tuple in zip(*stringed_arrs)])

def start_tablestr(title, col_titles):
    latex = [r"\begin{tabular}{|" + len(col_titles)*r"c|" + r"} "]
    latex.append(r" \hline ")
    latex.append(r"\multicolumn{" + str(len(col_titles)) + r"}{|c|}{\B{" + title + r"}} \\ ")
    latex.append(2*r" \hline ")
    latex.append(r" & ".join(map(lambda s: r"\emph{" + s + r"}", col_titles)))
    latex.append(r" \\ \hline ")
    return ''.join(latex)

end_tablestr = r" \\ \hline \end{tabular} "

def tabularize_data(*data_arrs, **kwargs):
    latex = [start_tablestr(kwargs['title'], kwargs['col_titles'])]
    latex.append(tabularize_columns(*data_arrs))
    latex.append(end_tablestr)
    return ''.join(latex)

def typeset_answer(qty, **kwargs):
    digits = kwargs.get('digits', 5)
    return '{:Lx}'.format((qty.magnitude).n(digits=digits)*qty.units)
