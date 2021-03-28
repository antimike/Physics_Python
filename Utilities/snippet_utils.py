import sys
import vim

texMathZones = ['texMathZone' + x for x in ['A', 'AS', 'B', 'BS', 'C', 'CS',
'D', 'DS', 'E', 'ES', 'F', 'FS', 'G', 'GS', 'H', 'HS', 'I', 'IS', 'J', 'JS',
'K', 'KS', 'L', 'LS', 'DS', 'V', 'W', 'X', 'Y', 'Z', 'AmsA', 'AmsB', 'AmsC',
'AmsD', 'AmsE', 'AmsF', 'AmsG', 'AmsAS', 'AmsBS', 'AmsCS', 'AmsDS', 'AmsES',
'AmsFS', 'AmsGS' ]] + ["VimwikiMath", "VimwikiEqIn"]
texIgnoreMathZones = ['texMathText']
texMathZoneIds = vim.eval('map('+str(texMathZones)+", 'hlID(v:val)')")
texIgnoreMathZoneIds = vim.eval('map('+str(texIgnoreMathZones)+", 'hlID(v:val)')")
ignore = texIgnoreMathZoneIds[0]
def is_math_mode():
    synstackids = vim.eval("synstack(line('.'), col('.') - (col('.')>=2 ? 1 : 0))")
    try:
        first = next(i for i in reversed(synstackids) if i in texIgnoreMathZoneIds or i in texMathZoneIds)
        return first != ignore
    except StopIteration:
        return False

def get_digits(s):
    return int(''.join(filter(str.isdigit, s)))

def substitute_text(text, snip):
    first = snip.snippet_start[0]
    last = snip.snippet_end[0]
    text[0] = snip.buffer[first][:snip.snippet_start[1]] + text[0]
    snip.buffer[last] = text[-1] + snip.buffer[last][snip.snippet_end[1] + 1:]
    snip.buffer = snip.buffer[:first] \
        + text[:-1] \
        + snip.buffer[last:]

_braces_lr = {r"(": r")", r"[": r"]", r"{": r"}", r"\{": r"\}"}
braces = {**_braces_lr, **{v: k for k, v in _braces_lr.items()}}
