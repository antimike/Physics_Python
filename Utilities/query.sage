"""test = {'a': lambda x: x + 1, 'b': {'c': 'boopy', 'd': lambda x: x^2}}
evaluate_query(3, test)"""
def evaluate_query(param, thunk, recalculate=False, **kwargs):
  if isinstance(thunk, dict):
    param_dict = param if isinstance(param, dict) else {}
    return {
      k: (not recalculate and param_dict.get(k)) or evaluate_query(param, v, recalculate)
      for k, v in thunk.items()
    }
  elif callable(thunk):
    return thunk(param)
  else:
    return thunk

def merge_query(param, thunk, recalculate=False):
  param |= evaluate_query(param, thunk, recalculate)

def merge_queries(params, thunk, **kwargs):
  for param in params:
    merge_query(param, thunk, **kwargs)
  if key := (kwargs.get('key') or thunk.get('key')):
    params.sort(key=key if callable(key) else lambda d: d.get(key))

def evaluate_queries(params, struct, **kwargs):
  """evaluate_queries.

  :param params: Parameters which will be fed to the query functions in 'struct.'
  :param struct: A dictionary of 'query functions' which will consume the params and (possibly) static members as well (i.e., non-callables).  A separate dict will be generated for each param; static members will simply be appended to each dict.
  :param kwargs: General catchall to allow 'kwargs chaining'.  Only looks specifically for 'key', which if supplied results in a sorted resultset.  The key may be provided by either (a) including it in the struct, or (b) including it in the kwargs; the latter always supercedes the former.  Furthermore, either a callable or non-callable key may be provided.  If callable, it will be passed to the sort function; if not, it will be used to index into the results to obtain the property to compare.
  """
  ret = [evaluate_query(param, struct, **kwargs) for param in params]
  if key := (kwargs.get('key') or struct.get('key')):
    ret.sort(key=key if callable(key) else lambda d: d.get(key))
  return ret

"""
test1 = {'a': {'A': 1, 'D': 2, 'E': 3}, 'b': {3, 4, 5}}
test2 = {'a': {'A': 4, 'D': 2, 'B': 5, 'C': 6}, 'c': 4}
deep_merge(test1, test2)
"""
def deep_merge(s1, s2):
  if isinstance(s1, dict):
    if isinstance(s2, dict):
      return {k: v for k, v in s1.items() if k in s1.keys() - s2.keys()} \
        | {k: v for k, v in s2.items() if k in s2.keys() - s1.keys()} \
        | {k: deep_merge(s1[k], s2[k]) for k in s1.keys() & s2.keys()}
    else:
      return s1 | {s2: None}
  elif isinstance(s2, dict):
    return s2 | {s1: None}
  else:
    return {s1, s2} if not s1 == s2 else s1

def deep_intersect(s1, s2):
  if isinstance(s1, dict) and isinstance(s2, dict):
    return {k: deep_intersect(s1[k], s2[k]) for k in s1.keys() & s2.keys()}
  else:
    return s1

def _deep_get(struct, key, eager=True, permissive=False):
  def _wrap(fn, perm):
    def wrapper(*args):
      try:
        return fn(*args)
      except:
        return perm
    return wrapper
  ret = []
  if isinstance(struct, dict):
    test = _wrap(key, permissive) if callable(key) else lambda k, v: k == key
    ret += [v for k, v in struct.items() if test(k, v)]
    for v in struct.values():
      if eager and not len(ret) == 0:
        break
      else:
        ret += _deep_get(v, key, eager)
  return ret

"""
test = {'a': {'b': {'c': 1, 'd': 2}, 'e': {'a': 3, 'b': 6, 'c': 7}}, 'f': 4}
deep_get(test, 'c', eager=False)
deep_get(test, lambda k, v: v < 6)
"""
def deep_get(struct, key, eager=True):
  ret = _deep_get(struct, key, eager)
  return (ret[0] if ret else None) if eager else ret
