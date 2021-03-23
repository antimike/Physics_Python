import sys

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
