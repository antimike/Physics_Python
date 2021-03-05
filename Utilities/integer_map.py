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

def iterates(function):
    """Returns an infinite iterator containing all the iterates of fn"""
    ret = function
    while True:
        yield ret
        ret = lambda x: function(ret(x))

def orbit(val, fn, stop_condition=lambda x: False):
    """Infinite iterator containing the orbit of `val` under application of `fn`"""
    history = {curr := val}
    while (new := fn(curr := val)) not in (history := history | {val}) and not stop_condition(new):
        yield (curr, val := new)[0]
    yield curr

# TESTS

def hailstone_fn(n):
    return n//2 if n%2 == 0 else 3*n + 1

def print_hailstone_graph(n):
    for m in orbit(n, hailstone_fn):
        print(m*'X')

for n in orbit(9, hailstone_fn):
    print(n)

print_hailstone_graph(19)

def test_unscramble(seq, pred, debug=False, max_iterations=100):
    max_iterations = min(max_iterations, len(seq))
    pred, seq = lambda x:x >= 4, [*range(1, 27)]
    images = orbit(
        scramble(pred, seq),
        lambda x: unscramble(pred, x, debug=debug),
        stop_condition=lambda x: x == seq
    )
    def get_msg(num, result):
        msg = (initial := 'Original: ') and not num or '<{:d}>: '
        return msg.format(num).center(len(initial)) + str(result)
    for msg in [get_msg(i, x) for i, x in enumerate(images) if i < max_iterations]:
        print(msg)




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
