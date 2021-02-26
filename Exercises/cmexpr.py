# Problem 'Complex Expressions (CMEXPR)' from SPOJ
# Goal: Given an input line consisting of letters (variables), parentheses,
# and arithmetic operators, return the same expression with all unnecessary 
# parentheses removed.

# Consistency check: 'last_operation_stack' should have the same number of elements
# as 'nesting_level', EXCEPT in one (very important) case:
# Immediately after parsing a ')', the 'last_operation_stack' can't be popped until
# the next operator is consumed by the parser, but the 'nesting_level' will be 
# decremented.  Thus, if 'nesting_level' = |'last_operation_stack'| - 1, then we
# know the next token has to be an operator which will determine whether the previous
# '()' pair can be stripped.
group_state = {
    'nesting_level': 0,
    'current_operator': '',
    'last_operation_stack': [],
    'pending_group_indices': []
}

def process_string(input):
    for char in input:
        if char == '(':
            group_state['nesting_level'] += 1
        elif char == ')':
            group_state['nesting_level'] -= 1
        elif char == '+' or char == '-':
            group_state['current_operator'] = char
        elif char == '*' or char == '/':
            group_state['current_operator'] = char
        else:
            assert True, "You done goofed"



