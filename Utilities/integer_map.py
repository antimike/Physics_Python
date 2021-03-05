"""
How to invert a boolean sort (subset selection) of a list:
first = [x for test(x) and x in l]
second = [x for not test(x) and x in l]
reconstructed = first + second
original = (reverse + split(test) + reverse)(reconstructed)
"""

"""
Explanation: "Boolean sort" specification of subsets / integer maps
Imagine a Boolean function guaranteed to return a result, say P (predicate).
Given a collection `list`, we can form `list1 = filter(P, list)` and `list2 = filter(not P, list)`.
We can use this construction to do several things:

1. Most obviously, we can think of `list1` as a subset of `list` parameterized by P.
So any function that calls for a map `S::Set -> S'::Set : S.contains(S')` can be passed P instead.

2. Alternatively, we can concatenate the two lists: `scrambled = list1 + list2`.
The result is formally a permutation `sigma` of the set `range(1, len(list))`.
In particular, we can ask what the inverse of `sigma` is in terms of P.
More interestingly:  Can this or similar constructions be used to parameterize *all* permutations?
Is there a reasonable subset of S_N that admits a particularly "nice" description in terms of such operators?
"""

def scramble(predicate, args):
    selected = [x for i, x in enumerate(args) if predicate(i)]
    unselected = [x for i, x in enumerate(args) if not predicate(i)]
    return [*selected, *unselected]


def unscramble(predicate, args, debug=False):
    args.reverse()
    if debug:
        print('-----' + str(args))
    args = scramble(predicate, args)
    if debug:
        print('-----' + str(args))
    args.reverse()
    if debug:
        print('-----' + str(args))
    return args


def test_unscramble(q, predicate, debug=False, max_iterations=100):
    print(q)
    qs = scramble(predicate, q)
    iterations = 1
    while (not qs == q) and iterations < min(len(q), max_iterations):
        print(qs)
        qs = unscramble(predicate, qs, debug=debug)
        iterations += 1
    if qs == q:
        print('It took {:d} iteration(s) to reconstruct the original list.'.format(iterations))
    else:
        print('Failed to reconstruct the original list.  Ran for {:d} iterations.'.format(iterations))
        print('Final state:')
        print('{:s}'.format(str(qs)))



test_unscramble([*range(5)], lambda x: x == 0 or x == 3, debug=True)

def test_slice(total_length, slice_length):
    q = [*range(total_length)]
    fn = lambda x: x >= slice_length
    test_unscramble(q, fn, debug=True)


test_fns = {
    'modular square': lambda x: x**2%27 < 10,
    'even': lambda x: x%2 == 0,
    'divisible by 7': lambda x: x%7 == 0,
    'prime': lambda x: len([y for y in range(2, x - 1) if x%y == 0]) == 0
}

for key, fn in test_fns.items():
    print('', '------------------------------------------')
    print('Testing predicate `{:s}`:'.format(key))
    test_unscramble([*range(28)], fn, debug=True, max_iterations=2)
