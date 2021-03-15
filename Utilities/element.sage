import mendeleev as chem
"""
TODOs:
    * Inject ureg through decorator
    * Consolidate method decorators into class decorators
    * Move calculated fields into either properties or descriptors 
    * Class methods / statics only where appropriate
    * Use inheritance to separate generic logic from source-specific methods
"""
random = oo
class Element:
    @staticmethod
    def get_wavelength(qty):
        if qty.check('[length]'):
            return qty
        elif qty.check('[frequency]'):
            return self._.Quantity(2)*pi.n(digits=5)*self._.c/qty
        elif qty.check('[energy]'):
            return Element.get_wavelength(qty/self._.hbar)
        else:
            raise ValueError
    @staticmethod
    def get_frequency(qty):
        if qty.check('[length]'):
            return self._.Quantity(2)*pi.n(digits=5)*self._.c/qty
        elif qty.check('[frequency]'):
            return qty
        elif qty.check('[energy]'):
            return qty/self._.hbar
        else:
            raise ValueError
    @staticmethod
    def _get_rms_n_values(n_squared):
        rms_v = (.5*(n_squared.abs() + 1))**.5
        rms_b = (.5*(n_squared.abs() + n_squared.magnitude.real()))**.5
        return (rms_v, rms_b)
    @staticmethod
    def reflectivity(n_squared):
        (rms_v, rms_b) = Element._get_rms_n_values(n_squared)
        return (rms_v**2 - rms_b)/(rms_v**2 + rms_b)
    @staticmethod
    def phase_tangent(n_squared):
        (rms_v, rms_b) = Element._get_rms_n_values(n_squared)
        return (n_squared.abs() - rms_b**2)**.5/(n_squared.abs() - rms_v**2)
    def _get_frequency_arg(fn):
        def wrapper(pointer, arg):
            return fn(pointer, Element.get_frequency(arg))
        return wrapper
    def __init__(self, symbol, ureg=None, **kwargs):
        self._ = ureg
        self.symbol = symbol
        self.mass = getattr(chem, self.symbol).mass*self._.gram/self._.mole
        self.f_0 = self._.Quantity(kwargs.get('f_0', 1))
    def lookup_data(self, **kwargs):
        self.mass_density = kwargs['densities'].get(self.symbol, None)
        self.strengths = [self._.Quantity(strength) for strength in kwargs['strengths'].get(self.symbol, None)]
        self.lifetimes = [lifetime/self._.hbar for lifetime in kwargs['lifetimes'].get(self.symbol, None)]
        self.resonances = [frequency/self._.hbar for frequency in kwargs['resonances'].get(self.symbol, None)]
        self.sigma_0 = kwargs['dc_conductivities'].get(self.symbol, None)
        return self
    def calculate_derived_properties(self):
        self.number_density = self.mass_density*self._.N_A/self.mass
        self.omega_c = (self.number_density*self._.e**2/(self._.epsilon_0*self._.m_e))**.5
        self.lambda_c = Element.get_wavelength(self.omega_c)
        self.gamma_0 = self.f_0*self.omega_c**2*self._.epsilon_0/self.sigma_0
        max_solns = [n(soln.rhs()) for soln in solve(Element.phase_tangent(self.epsilon_fr(x*self._.Hz))==.1, x) \
                     if soln.rhs() in RR]
        self.omega_max = oo if len(max_solns) == 0 else max_solns[0]*self._.Hz
        self.lambda_min = 0 if len(max_solns) == 0 else Element.get_wavelength(self.omega_max)
        return self
    @_get_frequency_arg
    def epsilon_fr(self, omega):
        return self._.Quantity(1) + self.strengths[0]*self.omega_c**2*I/(omega*(self.lifetimes[0] - omega*I))
    @_get_frequency_arg
    def epsilon_br(self, omega):
        bound_contributions = [self.strengths[j]*self.omega_c**2/(self.resonances[j]**2-omega**2-omega*self.lifetimes[j]*I)
                               for j, s in enumerate(self.strengths) if j > 0]
        return sum(bound_contributions)
    @_get_frequency_arg
    def reflectivity_drude(self, omega):
        return Element.reflectivity(self.epsilon_fr(omega))
    @_get_frequency_arg
    def reflectivity_full(self, omega):
        return Element.reflectivity(self.epsilon_fr(omega) + self.epsilon_br(omega))
