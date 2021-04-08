from abc import *
from functools import abstractmethod
from collections import namedtuple

def _debug(func_name='', msgs={}):
    _debugger = Debugger(func_name, msgs)
    def __debug(fn):
        def wrapped(*args, **kwargs, debug=False, debug_level=0):
            _debugger.enter(debug=debug, level=level, args=args, kwargs=kwargs)
            ret = fn(*args, **kwargs, debugger=_debugger)
            _debugger.end(ret)
    return wrapped
return __debug(fn)

class Debugger():
    def __init__(self):
        pass
    @abstractmethod
    def run(self):
        pass

"""
Plan:

"""
# Experiment: Class decorators



# Previous attempts
class Debugger():
    def _out(self, **kwargs):
        for msg, params in kwargs.items():
            msg = self._debug_msgs.get(msg)
            if msg is None:
                print('Debug message with key {} not found!'.format(str(msg)))
            elif callable(msg):
            try:
                print(msg(params))
            except:
                print('Your debug message "{}" failed to print with params "{}"'.format(str(msg), str(params)))
            else:
                print(str(msg) + str(params))

          @property
          def context(self):
              return self._debug_action if self._debug else self._do_nothing
  def _prepare_state(self, args, kwargs, debug_level):
      if self._debug:
          self._msg_count = 0
          def enter(self, debug=False, debug_level=0, args=None, kwargs=None):
              self._debug = debug
    self._debug_level = debug_level
    self._prepare_state(args, kwargs, debug_level)
    def end(self, return_val):
        self._print_msg('return')
        def __init__(self, func_name, **kwargs):
            self._func_name = func_name
    self._debug = False
    self._actions = kwargs
    def _perform_debug_action(self, action):
        if not self._debug:
            return
  def create_namedtuple(d):
      for k, v in d.items():

          class TestClass(object):
              pass
obj = TestClass()
obj.__dict__ |= {'a': 1, 'b': 2}
obj.a

@debug(out='{} {}', recurse={'increment': lambda x: x + 1, 'decrement': lambda x: x + 1})
state(recursion_depth.increment())
