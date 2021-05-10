
"""
Idea: Sets are predicates.
Given a function F: S --> T, and a subset T' < T---i.e., a predicate B: T --> {0, 1}---we can define the preimage of T' as the predicate S': s |--> B(F(s)).
The kernel of F is the subset of the domain K: S --> {0, 1} such that (B*F)(K(s)) == 0 for all s in S, where "B*F" is some (yet unspecified) mapping of F onto the set of Booleans (due to, e.g., a "truthy / falsey" valuation on T).
This is all rather obvious...
Preimage(fn, t) := set(filter(lambda s: f(s) == t, S))
{s: Preimage(fn, s) for s in S}
(s, t) --> (lambda s': s' == s, lambda t': t' == t) vs.
(s, t) --> lambda s', t': s' == s && t' == t
-----------------------------------------------------------
To each mapping f we can associate a function F: t |--> Preimage(t), no matter if f is well-defined (i.e., a "proper function") or not.
FF: F |--> f is constructed in the obvious way; F is then in the preimage of f under FF.
FF is "list destructuring."  So F---the "flattened dictionary"---is the primage of f (the "concatenated dict") under list destructuring.
The key is that preimages can be easily defined *functionally.*

"""

class Map:
    """Map.
    A kind of generalized `dict`, allowing a key to map to multiple values.
    Subset of the Cartesian product of two sets A and B.
    """
    def __init__(self, d, r):
        self._domain, self._range = d, r
    def maps(self):
        return lambda t: t in self._domain
    def maps_to(self):
        return lambda t: t in self._range

def Preimage(t, mapping):
    # return filter(...)
    pass

"""
(lambda item: item[0], lambda item: item[1])
(k, v) --> lambda d: {**d, d.get(k, {}).append(v)}
_d*d = {k: F(v)(v) for k, v in d.items()}


"""


def group_by(fn, collection):
    """group_by.
    Groups a collection by a given "property" `fn`
    `fn` must be callable and return a hashable value

    :param fn: Callable which supplies values to group items by
    :param collection: Items to group

    >>> l = [('a', 1), ('a', 2), ('b', 2), ('c', 3)]
    >>> group_by(lambda p: p[0], l)
    """
    if not callable(fn):
        try:
            raise ValueError("Cannot group by non-callable property '{}'".format(fn.__name__))
        except AttributeError:
            raise ValueError("Argument '{}' is not callable and has no '__name__' attribute".format(fn))
    return {
        image: list(filter(lambda s: fn(s) == image, collection))
        for image in set(map(fn, collection))
    }
def _merge_dicts(dict_list):
    """_merge_dicts.
    Merges a list of `dicts`: If k in d1 and k in d2, then the merged dict M has M[k] = [d1[k], d2[k]]

    :param dict_list: List of `dicts` to merge

    >>> d1, d2 = {'a': 1, 'b': 2}, {'a': 2, 'c': 3}
    >>> _merge_dicts([d1, d2])
    >>> {'a': [1, 2], 'b': [2], 'c': [3]}
    """
    return {
        key: [d.get(key) for d in dict_list if key in d]
        for key in set().union(*list(d.keys() for d in dict_list))
    }

