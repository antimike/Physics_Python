def stringify_array(arr):
    return map(lambda x: str(x), arr)

def convert_args(converter):
    def decorator(fn):
        def wrapper(*args, **kwargs):
            return fn(
                *[converter(arg) for arg in args],
                **{key: converter(val) for key, val in kwargs}
            )
        return wrapper
    return decorator

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

# Passes preliminary testing
class Latex_Serializer:
    text_transformations = {
        'bold': lambda str: r"\B{" + str + r"}",
        'italics': lambda str: r"\emph{" + str + r"}"
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
        self._opts = {**Latex_Serializer.datum_default_opts, **Latex_Serializer.text_default_opts, **kwargs}
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
            ret = str(datum)
        for key in Latex_Serializer.text_transformations.keys():
            if kwargs[key]:
                ret = Latex_Serializer.text_transformations[key](str)
        return ret
    @apply_defaults
    def serialize_text(self, string, **kwargs):
        if kwargs['parenthetical_units'] is not None:
            string += ' (' + '{:Lx}'.format(kwargs['parenthetical_units']) + ')'
        for key in Latex_Serializer.text_transformations.keys():
            if kwargs[key]:
                string = Latex_Serializer.text_transformations[key](string)
        return string
    @apply_defaults
    def serialize(self, arg, **kwargs):
        if kwargs['data']:
            self.serialize_datum(arg, **kwargs)
        if kwargs['text']:
            self.serialize_text(arg, **kwargs)



class Table:
    defaults = {
        'delimiter': '|',
        'placeholder': '-',
        'alignment': 'c',
        'pre_title': r" \hline ",
        'post_title': r" \hline ",
        'title_left_border': r"|",
        'title_alignment': r"c",
        'title_right_border': r"|",
        'pre_col_titles': r" \hline ",
        'post_col_titles': r" \\ \hline ",
        'left_border': '|',
        'right_border': '|'
    }
    @staticmethod
    def tabularize_rows(rows):
        return r" \\ ".join([r" & ".join(row) for row in rows])
    @staticmethod
    def title_col(title, num_rows):
        multirow = r"\multirow{" + str(num_rows) + r"}{*}{" + title + r"}"
        return [multirow] + ['']*(num_rows - 1)
    @staticmethod
    def title_row(title, num_cols):
        if num_cols > 0:
            multicol = r"\multicolumn{" + str(num_cols) + r"}{c}{" + title + r"}"
        else:
            multicol = ''
        return [multicol]
    @staticmethod
    def pad_arrs(arrs, min_length, placeholder='-'):
        l = max(min_length, max([len(arr) for arr in arrs]))
        return [arr + [placeholder]*(l - len(arr)) for arr in arrs]
    # @staticmethod
    # def pad_columns(arrs, num_cols, placeholder='-'):
        # num_rows = max([len(arr) for arr in arrs])
        # arrs = Table.pad_arrs(arrs, num_rows, placeholder)
        # return arrs + [[placeholder]*num_rows]*(num_cols - len(arrs))
    @staticmethod
    def transpose(arrs, min_length, placeholder='-'):
        arrs = Table.pad_arrs(arrs, min_length, placeholder=placeholder)
        length = max([len(arr) for arr in arrs])
        return [[arr[j] for arr in arrs] for j in range(length)]
    def __init__(self, **kwargs):
        self._serializer = Latex_Serializer(**kwargs)
        self._num_cols = 0
        self._opts = {**kwargs, **Table.defaults}
        self._hlines = []
        self._col_structure = [self._opts['left_border']]
        self._col_titles = []
        self._rows = []
        self._has_row_titles = False
    @apply_defaults
    def add_columns(self, *cols, **kwargs):
        if 'title' in kwargs and len(cols) > 0:
            self._col_titles = Table.pad_arrs([self._col_titles], self._num_cols, placeholder='')[0]
            self._col_titles += Table.title_row(kwargs['title'], len(cols))
        old_cols = Table.transpose(self._rows, self._num_cols, placeholder=kwargs['placeholder'])
        old_cols += cols
        self._rows = Table.transpose(old_cols, len(self._rows), placeholder=kwargs['placeholder'])
        self._num_cols += len(cols)
        self._col_structure += [kwargs['alignment'] + kwargs['delimiter']]*len(cols)
    @apply_defaults
    def add_rows(self, *rows, **kwargs):
        cols = Table.transpose(rows, self._num_cols, placeholder=kwargs['placeholder'])
        if 'title' in kwargs:
            cols.insert(0, Table.title_col(kwargs['title'], len(rows)))
            if not self._has_row_titles:
                self._add_title_column()
            self._has_row_titles = True
        elif self._has_row_titles:
            cols.insert(0, ['']*len(rows))
        self._num_cols = max(len(cols), self._num_cols)
        self._rows += Table.transpose(cols, len(rows), placeholder=kwargs['placeholder'])
    @serialize
    @apply_defaults
    def add_column_titles(self, col_titles, **kwargs):
        self._col_titles = [Table.title_row(*pair) for pair in col_titles]
    def add_vline(self):
        self._col_structure.append('|')
    def add_hline(self):
        self._hlines.append(len(self._rows))
    def _add_title_column(self):
        placeholder = self._opts['placeholder']
        cols = Table.transpose(self._rows, self._num_cols, placeholder=placeholder)
        cols.insert(0, ['']*len(self._rows))
        self._rows = Table.transpose(cols, self._num_cols, placeholder=placeholder)
    def _pad_rows(self):
        self._rows = Table.pad_arrs(
            self._rows,
            self._num_cols,
            placeholder=self._opts['placeholder']
        )
    def _add_hlines(self):
        positions = self._hlines
        positions.sort(reverse=True)
        rows = self._rows
        hline = r" \hline "
        for pos in positions:
            rows.insert(pos, [hline])
        return rows
    def _start_table(self):
        latex = [r"\begin{tabular}{|" + ''.join(self._col_structure) + r"} "]
        latex.append(self._opts['pre_title'])
        return ''.join(latex)
    def _table_title(self):
        latex = [
            r"\multicolumn{" \
            + str(self._num_cols) \
            + r"}{" \
            + self._opts['title_left_border'] \
            + self._opts['title_alignment'] \
            + self._opts['title_right_border'] \
            + r"}{\B{" \
            + self._opts['title'] \
            + r"}} \\ "
        ]
        latex.append(self._opts['post_title'])
        return ''.join(latex)
    def _table_col_title_row(self):
        latex = [self._opts['pre_col_titles']]
        latex.append(r" & ".join(self._col_titles))
        latex.append(self._opts['post_col_titles'])
        return ''.join(latex)
    def _end_table(self):
        return r" \\ \hline \end{tabular} "
    @property
    def latex(self):
        self._pad_rows()
        rows = self._add_hlines()
        return self._start_table() \
            + self._table_title() \
            + self._table_col_title_row() \
            + Table.tabularize_rows(rows) \
            + self._end_table()

# Tasks remaining:
# Add "hrule" and "vrule" options
# Enforce serialization before other logic through decorators
# Write Text and Math serializers and finish Data

def serialize_first(fn):
    # Serializers: Data, Math, Text
    pass

def apply_defaults(fn):
    def wrapper(obj, *args, **kwargs):
        return fn(obj, *args, **{**obj.defaults, **kwargs})
    return wrapper

def serialize(fn):
    def wrapper(obj, arg, **kwargs):
        return fn(obj, obj._serializer.serialize(arg, **kwargs), **kwargs)
    return wrapper
