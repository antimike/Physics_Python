import latex_serializer as tex

class Table:
    defaults = {
        'delimiter': '|',
        'placeholder': '-',
        'alignment': 'c',
        'title_opts': {
            'data': False,
            'text': True,
            'bold': True,
            'italics': False
        },
        'pre_title': r" \hline ",
        'post_title': r" \hline ",
        'title_left_border': r"|",
        'title_alignment': r"c",
        'title_right_border': r"|",
        'col_title_opts': {
            'data': False,
            'text': True,
            'bold': False,
            'italics': True
        },
        'pre_col_titles': r" \hline ",
        'post_col_titles': r" \\ \hline ",
        'row_title_opts': {
            'data': False,
            'text': True,
            'bold': True,
            'italics': False
        },
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
    def title_row(title, num_cols, left_delimiter='|', right_delimiter = '|'):
        if num_cols > 0:
            multicol = r"\multicolumn{" \
                + str(num_cols) \
                + r"}{" \
                + left_delimiter \
                + r"c" \
                + right_delimiter \
                + r"}{" + title + r"}"
        else:
            multicol = ''
        return multicol
    @staticmethod
    def pad_arrs(arrs, min_length, placeholder='-'):
        if len(arrs) == 0:
            return arrs
        l = max(min_length, Table._safe_max_length(arrs))
        return [arr + [placeholder]*(l - len(arr)) for arr in arrs]
    @staticmethod
    def transpose(arrs, min_length=0, placeholder='-'):
        arrs = Table.pad_arrs(arrs, min_length, placeholder=placeholder)
        length = Table._safe_max_length(arrs)
        return [[arr[j] for arr in arrs] for j in range(length)]
    @staticmethod
    def _safe_max_length(arrs):
        if arrs is None or len(arrs) == 0:
            return 0
        else:
            return max([len(arr) for arr in arrs])
    def __init__(self, **kwargs):
        self._serializer = tex.Latex_Serializer(**kwargs)
        self._num_cols = 0
        self._opts = {**kwargs, **Table.defaults}
        self._hlines = []
        self._col_structure = []
        self._col_titles = []
        self._rows = []
        self._has_row_titles = False
    # TODO: Fix the way this is padding the column titles list (doesn't properly account for multicols)
    @tex.serialize_args
    @tex.apply_defaults
    def add_columns(self, *cols, **kwargs):
        if 'col_title' in kwargs and len(cols) > 0:
            col_title_opts = {**kwargs, **kwargs['col_title_opts']}
            self._col_titles = Table.pad_arrs([self._col_titles], self._num_cols, placeholder='')[0]
            self._col_titles.append(
                Table.title_row(
                    *self._serializer.serialize(kwargs['col_title'], **col_title_opts),
                    len(cols),
                    left_delimiter=''
                )
            )
        old_cols = Table.transpose(self._rows, self._num_cols, placeholder=kwargs['placeholder'])
        old_cols += cols
        self._rows = Table.transpose(old_cols, len(self._rows), placeholder=kwargs['placeholder'])
        self._update_num_cols(lambda x: x + len(cols), **kwargs)
    @tex.serialize_args
    @tex.apply_defaults
    def add_rows(self, *rows, **kwargs):
        cols = Table.transpose(rows, self._num_cols, placeholder=kwargs['placeholder'])
        # NOTE: This call has to occur before the 'row_title' logic, since the row title adds a spurious increment to the column count
        self._update_num_cols(lambda x: max(len(cols), x), **kwargs)
        if 'row_title' in kwargs:
            row_title_opts = {**kwargs, **kwargs['row_title_opts']}
            cols.insert(
                0,
                Table.title_col(
                    *(self._serializer.serialize(kwargs['row_title'], **row_title_opts)),
                    len(rows)
                )
            )
            if not self._has_row_titles:
                self._add_title_column()
            self._has_row_titles = True
        elif self._has_row_titles and len(rows) > 0:
            cols.insert(0, ['']*len(rows))
        self._rows += Table.transpose(cols, len(rows), placeholder=kwargs['placeholder'])
    @tex.serialize(lambda pairs: [t for t, n in pairs],
               lambda titles, pairs: [(t, n) for t, (_, n) in zip(titles, pairs)])
    # @apply_defaults
    # Unfortunately, the order in which the default kwargs are merged with the passed kwargs is important here and does not agree with the convention used in other class functions
    def add_column_titles(self, *col_titles, **kwargs):
        self._col_titles = [Table.title_row(*pair, left_delimiter = '') for pair in col_titles]
        num_cols = sum([n for t, n in col_titles])
        self._update_num_cols(lambda x: max(x, num_cols), **kwargs)
        if self._num_cols > num_cols:
            self._col_titles = Table.pad_arrs([self._col_titles], self._num_cols, placeholder='')
    def add_vline(self):
        self._col_structure.append('|')
    def add_hline(self):
        self._hlines.append(len(self._rows))
    # Shouldn't need to apply defaults to private methods
    def _update_num_cols(self, update_fn, **kwargs):
        result = update_fn(self._num_cols)
        delm = kwargs.get('delimiter', self._opts['delimiter'])
        align = kwargs.get('alignment', self._opts['alignment'])
        change = result - self._num_cols
        if self._num_cols > 0 and change > 0:
            self._col_structure.append(delm)
        self._col_structure.append(delm.join(align*change))
        self._num_cols = result
    def _add_title_column(self):
        placeholder = self._opts['placeholder']
        cols = Table.transpose(self._rows, self._num_cols, placeholder=placeholder)
        cols.insert(0, ['']*len(self._rows))
        self._rows = Table.transpose(cols, self._num_cols, placeholder=placeholder)
        self._col_structure.insert(0, self._opts['alignment'] + self._opts['delimiter'])
    def _pad_rows(self):
        self._rows = Table.pad_arrs(
            self._rows,
            self._num_cols,
            placeholder=self._opts['placeholder']
        )
    def _add_hlines(self):
        #positions = self._hlines
        #positions.sort(reverse=True)
        #rows = self._rows
        #hline = r" \hline "
        #for pos in positions:
            #rows.insert(pos, [hline])
        #return rows
        hline = r" \hline "
        for pos in self._hlines:
            self._rows[pos][-1] += hline
    def _start_table(self):
        latex = [
            r"\begin{tabular}{" \
            + self._opts['left_border'] \
            + ''.join(self._col_structure) \
            + self._opts['right_border'] \
            + r"} "
        ]
        latex.append(self._opts['pre_title'])
        return ''.join(latex)
    def _table_title(self):
        latex = [
            r"\multicolumn{" \
            + str(self._num_cols if not self._has_row_titles else self._num_cols + 1) \
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
        if self._has_row_titles:
            latex.append(r" & ")
        latex.append(r" & ".join(self._col_titles))
        latex.append(self._opts['post_col_titles'])
        return ''.join(latex)
    def _end_table(self):
        return r" \\ \hline \end{tabular} "
    @property
    def title(self):
        return self._opts['title']
    @title.setter
    def title(self, val):
        self._opts['title'] = val
    @property
    def latex(self):
        self._pad_rows()
        rows = self._add_hlines()
        return self._start_table() \
            + self._table_title() \
            + self._table_col_title_row() \
            + Table.tabularize_rows(rows) \
            + self._end_table()
