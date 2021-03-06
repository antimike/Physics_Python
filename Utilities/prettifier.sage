# Credit to dsejas on ask.sagemath.org
# Link to question: https://ask.sagemath.org/question/56447/is-there-a-way-to-specify-that-sqrts-should-be-avoided-in-favor-of-exponents-when-using-the-latex-function/

from sage.symbolic.operators import arithmetic_operators as _ops # list of predefined arithmetic operators

# THIS FUNCTION WILL COMPUTE THE BASE AND EXPONENT OF A LIST OF EXPRESSIONS:
def _base_exp(exprs):
    res = [] # this will contain the expressions in "exprs", but with their respective exponent
    for ex in exprs:
        op = ex.operator() # obtain the operator of ex
        if (op==None) or (_ops[op]!='^'): # If the "ex" is a simple expression (op == None) or not a power,...
            res.append([ex, 1]) # ...then the exponent is 1;...
        else: # if the operator is that of a power,...
            res.append(ex.operands()) # ...then extract the base and the exponent
    return res

# THIS FUNCTION TAKES CARE OF "PRETTIFYING" A PRODUCT, GROUPING BASES WITH COMMON EXPONENTS:
def _prettify_prod(terms, prod_symbol=r'\,'):
    n = len(terms)
    base_exp = _base_exp(terms) # obtain a list of terms in the product with their respective exponents
    res = ''
    con = [] # this will store the already consumed exponents
    for i in range(n):
        (b, e) = base_exp[i] # obtain the i-th term and its exponent
        aux = [] # this will store terms with the same exponent
        if e not in con: # If "e" is not an already consumed exponent,...
            con.append(e) # ...then add it to the list of consumed exponents.
            for j in range(i, n):
                (b1, e1) = base_exp[j] # obtain the j-th term and its exponent
                if e1 == e: # If the j-th term has the exponent we are consuming,...
                    aux.append(prettify(b1)) # ...then add the prettified term to the list.
            aux1 = prod_symbol.join(aux) # multiply the terms in the list (this is customizable with the "prod_symbol" option)
            if len(aux) > 1: # If there are more than one term with the same exponent, ...
                aux1 = r'\left(' + aux1 + r'\right)' # ...then surround them with parenteses.
            if e == 1: # If the exponent we are currently consuming is 1,...
                res = aux1 + prod_symbol + res # ...then this must be the first term in the product, so we add it to the front;...
            else: # ...for any other exponent,...
                res += aux1 + '^{' + prettify(e) + '} ' # we put the prettified exponent in place and append the resulting term to the product.
    return res

# THIS IS THE MAIN (RECURSIVE) FUNCTION
def prettify(expr, prod_symbol=r'\,'):
    if (not hasattr(expr,'operands')) or (expr.operator()==None): # If this is a simple expression not needing special treatment,...
        return latex(expr)# ...then return the corresponding LaTeX code.
    terms = expr.operands() # extract the terms that make up this expression
    op = _ops[expr.operator()] # extract the main operator of this expression in text form
    res = ''
    if op == '*': # If the operator is multiplication,...
        res += _prettify_prod(terms, prod_symbol) # ...then let _prettify_prod() take care of it;...
    else: # ...otherwise,...
        for term in terms:
            res += prettify(term, prod_symbol) + op # ...then recursively call prettify() on each term and append the operator (this will produce an extra operator.)
    return res[:len(res)-1] # Since there is an extra operator, return all, except the last character.

def prettifier(prod_symbol=r'\,'):
    return lambda expr: prettify(expr, prod_symbol)

#TODO Add facilities for dealing with units
#expr = 3/7*sqrt(2)*sqrt(pi)*(5)^(8/7) + 2
#show(LatexExpr(prettify(expr)))
