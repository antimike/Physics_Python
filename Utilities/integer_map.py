def scramble(predicate, args):
    selected = [x for i, x in enumerate(args) if predicate(i)]
    unselected = [x for i, x in enumerate(args) if not predicate(i)]
    return [*selected, *unselected]


def unscramble(predicate, beep_boop, debug=False):
    args = beep_boop
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
