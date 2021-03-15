"""
Problems:
   - "Default opts" logic
       - Different for different functions
       - Generally handled by reducers
       - Differences are generally limited to composition order
   - "Context" logic (e.g., loading data)
   - Preprocessing arguments, fetched data, and defaults
       - e.g., units
   - "General function" definitions:
       - Which should be declared static and which should just be declared as lone functions?

Possible solutions:
   - Inheritance
   - Decorators:
       - Function decorators (e.g., reducers)
       - Class decorators
   - Managed attributes:
       - Properties
       - Descriptors
       - @cached_property
   - Class methods @classmethod:
       - Can dynamically update (applicable to all instances!)
"""
# class preprocessor:
    # default_reducers = [
    # ]
    # @staticmethod
    # def 
    # def __call__(self, *args, **kwargs):
        # pass
    # def __init__(self, reducers=None):


def apply_defaults(fn):
    def wrapper(obj, *args, **kwargs):
        return fn(obj, *args, **{**obj._opts, **kwargs})
    return wrapper

def reduce_data_opts(table, opts):
    return {**table._opts, **opts}

def reduce_col_title_opts(table, opts):
    data_kwargs = reduce_data_opts(table, opts)
    col_title_opts = {**table._opts['col_title_opts'], **opts, **opts['col_title_opts']}
    return {**reduce_data_opts(table, opts), 'col_title_opts': col_title_opts}

def serialize_args(fn):
    def wrapper(obj, *args, **kwargs):
        return fn(obj, *(obj._serializer.serialize(*args, **kwargs)), **kwargs)
    return wrapper

# def serialize_args(serialize_from_args, args_from_serialized, opts_from_kwargs):
    # def decorator(fn):
        # def wrapper(instance, *args, **kwargs):
            # return fn(
                # instance,
                # *args_from_serialized(instance._serializer.serialize(*serialize_from_args(args), **opts_from_(kwargs)), args),
                # **kwargs
            # )
        # return wrapper
    # return decorator

def serialize_kwargs(projection, inclusion):
    def decorator(fn):
        def wrapper(instance, *args, **kwargs):
            return fn(
                instance,
                *args,
                **inclusion(instance._serializer.serialize(*projection(args), **kwargs), args),
                **kwargs
            )
        return wrapper
    return decorator

# - Preprocess:
    # 1. Get col_title_opts and row_title_opts; merge into kwargs
    # 2. Apply defaults

class Latex_Serializer:
    text_transformations = {
        'bold': lambda x: r"\B{" + str(x) + r"}",
        'italics': lambda x: r"\emph{" + str(x) + r"}"
    }
    serialization_type_defaults = {
        'text': True,
        'data': False
    }
    text_default_opts = {
        'bold': False,
        'italics': False,
        'parenthetical_units': None
    }
    datum_default_opts = {
        'transformation': lambda x: x,
        'units': None,
        'show_units': True,
        'digits': 5
    }
    def __init__(self, **kwargs):
        self._opts = {
            **Latex_Serializer.datum_default_opts,
            **Latex_Serializer.text_default_opts,
            **Latex_Serializer.serialization_type_defaults,
            **kwargs
        }
    @apply_defaults
    def serialize_datum(self, datum, **kwargs):
        datum = kwargs['transformation'](datum)
        ret = ''
        try:
            if kwargs['units'] is None:
                datum = datum.to_base_units()
            else:
                datum = datum.to(kwargs['units'])
            units = datum.units
            if not kwargs['show_units']:
                datum /= units
            ret = '{:Lx}'.format(n(datum.magnitude, digits=kwargs['digits'])*datum.units)
        except AttributeError:
            ret = str(n(datum, digits=kwargs['digits']))
        # for key in Latex_Serializer.text_transformations.keys():
            # if kwargs[key]:
                # ret = Latex_Serializer.text_transformations[key](ret)
        return ret
    @apply_defaults
    def serialize_text(self, arg, **kwargs):
        string = str(arg)
        if kwargs['parenthetical_units'] is not None:
            string += ' (' + '{:Lx}'.format(kwargs['parenthetical_units']) + ')'
        for key in Latex_Serializer.text_transformations.keys():
            if kwargs[key]:
                string = Latex_Serializer.text_transformations[key](string)
        return string
    @apply_defaults
    def serialize(self, *args, **kwargs):
        return [self._serialize(arg, **kwargs) for arg in args]
    def _serialize(self, arg, **kwargs):
        if isinstance(arg, list):
            return self.serialize(*arg, **kwargs)
        else:
            if kwargs['data']:
                arg = self.serialize_datum(arg, **kwargs)
            if kwargs['text']:
                arg = self.serialize_text(arg, **kwargs)
            return arg
