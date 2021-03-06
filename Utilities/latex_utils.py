from pint import UnitRegistry
from matplotlib import rc
import mendeleev as chem
# import vim

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








class Table:
    class Column:
        def __init__(self, ureg, none_char, show_units=True, show_title_units=False):
            self._ureg = ureg
            self._units = self._ureg.dimensionless
            self._title = ''
            self._data = []
            self._none_char = none_char
            self._show_units = show_units
            self._show_title_units = show_title_units
        @property
        def units(self):
            return self._units
        @property
        def title(self):
            return self._title

    def __init__(self):
        self._str = ''
        self._num_cols = 0
        self._num_rows = 0
        self._title = ''
    def add_columns(self, *cols, **kwargs):
        self._str += tabularize_columns(*cols)
    def add_rows(self, *rows, **kwargs):
        pass
    @property
    def latex(self):
        return self._str

def stringify_array(arr):
    return map(lambda x: str(x), arr)

@convert_args(stringify_array)
def tabularize_rows(*rows):
    return r" \\ ".join([r" & ".join(row) for row in rows])

@convert_args(stringify_array)
def tabularize_columns(*cols):
    return r" \\ ".join([r" & ".join(tuple) for tuple in zip(*cols)])

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

@accept_lists
@convert_args(str)
def bold(str):
    return r"\textbf{" + str + r"}"

@accept_lists
@convert_args(str)
def math(str):
    return r"$" + str + r"$"

def paragraphs_from_file(stream, delimiter='\n'):
    lines = []
    for line in stream:
        if line == delimiter and lines:
            yield lines
            lines = []
        else:
            lines.append(line.rstrip())
    yield lines

def get_cols(file):
    with open(file, mode='r') as f:
        return list(paragraphs_from_file(f))

def test_paragraph_serialization():
    filename = './testfile.txt'
    cols = get_cols(filename)

def accept_lists(fn):
    def wrapper(arg):
        if isinstance(arg, list):
            return [fn(elem) for elem in arg]
        else:
            return fn(arg)
    return wrapper

def convert_args(converter):
    def decorator(fn):
        def wrapper(*args, **kwargs):
            return fn(
                *[converter(arg) for arg in args],
                **{key: converter(val) for key, val in kwargs}
            )
        return wrapper
    return decorator

# Not sure if this is a rabbit-hole worth diving into
class expr:
    def __init__(self, *args):
        self._str = ''
    def bold(self):
        return r"\mathbf{" + self._str + r"}"

# texMathZones = ['texMathZone' + x for x in ['A', 'AS', 'B', 'BS', 'C', 'CS',
# 'D', 'DS', 'E', 'ES', 'F', 'FS', 'G', 'GS', 'H', 'HS', 'I', 'IS', 'J', 'JS',
# 'K', 'KS', 'L', 'LS', 'DS', 'V', 'W', 'X', 'Y', 'Z', 'AmsA', 'AmsB', 'AmsC',
# 'AmsD', 'AmsE', 'AmsF', 'AmsG', 'AmsAS', 'AmsBS', 'AmsCS', 'AmsDS', 'AmsES',
# 'AmsFS', 'AmsGS' ]] + ["VimwikiMath", "VimwikiEqIn"]
# texIgnoreMathZones = ['texMathText']
# texMathZoneIds = vim.eval('map('+str(texMathZones)+", 'hlID(v:val)')")
# texIgnoreMathZoneIds = vim.eval('map('+str(texIgnoreMathZones)+", 'hlID(v:val)')")
# ignore = texIgnoreMathZoneIds[0]
def is_math_mode(vim):
    return vim.eval('vimtex#syntax#in_mathzone()') == '1'
    # synstackids = vim.eval("synstack(line('.'), col('.') - (col('.')>=2 ? 1 : 0))")
    # try:
        # first = next(i for i in reversed(synstackids) if i in texIgnoreMathZoneIds or i in texMathZoneIds)
        # return first != ignore
    # except StopIteration:
        # return False
