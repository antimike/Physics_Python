

@debug(alias='db', test='{} is {}', level=1)
def test_fn(a, b, db=None):
    db.indent(2).test('this', 'awesome').level(1)
    print(str(a) + ' ' + str(b))

test_fn.active
test_fn('this', 'awesome', debug=True)
test_fn._wrapped('hello', 'world')

test_fn('hello', 'world', debug=True)
test_fn.active
Debugger._pass_predicate(test_fn)
dbg = Debugger(test_fn, test='{} is {}', level=1)
dbg.test('boopy', 'shadoopy').level()()
dbg.print_state('indent')()
dbg.level()()
dbg.indent(lambda s: lambda state: state['level'])
dbg.tab(10*'-')
dbg.indent()()
dbg.test('First', 'second', indent=1, tab='-----')()
dbg.test
