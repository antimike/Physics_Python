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
