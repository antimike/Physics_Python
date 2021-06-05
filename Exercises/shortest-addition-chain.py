""" Thoughts """

    # Different data structure:
    # Tree whose nodes are sums and whose edges are labeled with binary strings of length <= len(root_path)
    # Here, root_path is the (unique) path from the root node to the terminal node of the edge
    # Basically a breadth-first search
    # OK, final idea: The list of nums from "levels" that have been fully explored doesn't need any more structure.
    # In particular, you should never have to do anything but append to this list.
    # The temp list maintained while iterating over the next "level", however, will need to support sorted insertion.

from __future__ import annotations              # So type hints will work with class types inside class definitions
import logging
import time
from abc import abstractmethod, ABC
from typing import TypeVar, final, Type, Callable, Optional, Hashable, Any

logger = logging.getLogger()
INDENT = '    '
T = TypeVar('T')

class Updater(ABC):
    """ABC Updater.
    Model for a container which maintains mutable data and needs to update different parts of it in a controlled way.
    """

    @classmethod
    @abstractmethod
    def update_item(cls: Type[Updater], original: T, update: T) -> None:
        """Updater.update_item
        Updated the first argument by interpreting the second as the update.
        """
        if cls.is_atomic(type(original)):
            raise TypeError(
                "Updater {upd} cannot update type {tp}".format(
                    upd=cls, tp=type(original))
            )

    @final
    @classmethod
    def is_atomic(cls: Type[Updater], C: Type[T]) -> bool:
        """Updater.is_atomic
        Alias for `not Updater.is_updatable`.  Returns True if the argument is a type that cannot be updated by the calling class.
        """
        return not cls.is_updatable(C)

    @classmethod
    @abstractmethod
    def is_updatable(cls: Type[Updater], C: Type[T]) -> bool:
        """Updater.is_updatable
        Returns True if the argument is a type that can be updated by the calling class.
        """
        return NotImplemented

class Updatable(ABC):
    """ABC Updatable.
    Model for a container which can accept updates.
    """

    @classmethod
    def __subclasshook__(cls: Type[Updatable], C: Type[T]) -> bool:
        if cls is Updatable:
            if any("update" in Sup.__dict__ for Sup in C.__mro__):
                return True
        return NotImplemented

    @abstractmethod
    def update(self: Updatable, other: Updatable) -> None:
        pass

class State(Updater, Updatable):
    """ABC State.
    Model for a stateful, mutable data container.
    """

    @classmethod
    def update_item(cls: Type[State], first: T, second: T) -> None:
        """
        Raises an ArgumentException through the call to super() if called on non-Updatable objects.
        """
        super().update(first, second)
        first.update(second)

    def copy(self: State) -> None:
        return super().copy()

    @classmethod
    def is_updatable(cls: Type[State], C: Type[T]) -> bool:
        """Defines objects updatable by State containers: Other State containers."""
        return issubclass(C, Updatable)

    def __or__(self: State, other: State) -> State:
        """Enables use of `state |= update` idiom."""
        # TODO: Think of a better way to do this
        copy = self.copy()
        copy.update(other)
        return copy

class StateDict(dict, State):
    """class StateDict.
    A subclass of dict and State which recursively calls all substates' 'update' methods on update.
    """

    @final
    def __update__(self: StateDict, other: StateDict) -> None:
        for key in other:
            if key in self:
                self._update_item(key, other[key])
            else:
                self._add_item(key, other[key])

    def _update_item(self, key, val) -> None:
        try:
            self.__class__.update_item(self[key], val)
        except TypeError:
            self[key] = val

    def _add_item(self: StateDict, key: Hashable, val: Any = None) -> None:
        self[key] = val

    @final
    def update(self: StateDict, *args, **kwargs) -> None:
        self.__update__(
            self.__class__(*args, **kwargs)
        )

class View(StateDict):
    """class View.
    A State container whose keys are effectively immutable.
    In other words, it is only possible to update the keys that the View is initialized with.
    """

    def _add_item(self, key, val=None):
        pass

class History(list, Updatable):
    """class History.
    Simple container to remember state or computational information.
    Subclasses list and defines an 'update' method.
    """

    def update(self, other):
        self.extend(other)

    def push(self, item):
        self.update(item)

def shortest_addition_chain(target, seed=[1], state=None, prune_aggressively=True):
    """
    Observations:
    =============
    1. Number of binary digits of N is minimal for the star-chain problem
    2. Binary digit sum (popcount) plus number of digits is maximal

    Ideas:
    ======
    * Use popcount to transform the problem:
        * Can we look for N* such that PC(N*) = N?  The number of "binary bins" would then correspond to the chain size
        * If not popcount, the first row of the DFT, try other rows of the DFT...
        * What are the conditions on a valid N*?
        * Each N* has to be built out of number of the form 2**S*(2**m - 1):
            * S is the "shift width," i.e., how far to the left the island is
            * m is the size of the bin
    * Counting "binary bins"
    * Use DFT somehow?
        * Can we convert the additive requirement to a multiplicative one on roots of unity?
    """
    pass

def binary_bins(N):
    """
    Computes the number of "binary islands" or "bins" consisting of contiguous 1's in the binary expansion of N

    >>> nums = [1, 11, 121, sum([2**(2*k + 1) for k in range(10)])]
    >>> print(*map(binary_bins, nums))
    1 2 2 10
    """
    return popcount(N^(N<<1)) >> 1


def popcount(N):
    """
    Computes the number of 1's in the  binary expansion of N

    >>> popcount(8)
    1
    >>> popcount(2**15 + 2**13 + 2**12 + 2**11 + 2**9 + 2**6 + 2**5 + 2**3 + 2**2 + 2)
    10
    """
    count = 0
    while N:
        count += 1
        N &= N-1
    return count

def shortest_star_chain(target, seed=[1], container=None, prune_aggressively=True):
    """
    >>> shortest_star_chain(15)
    ([1, 2, 3, 6, 12, 15], 6)
    >>> # The following is a counterexample to the claim that an optimal chain must 
    >>> # always "use" its last element in the construction of the next one.
    >>> # (This example is due to Hansen, 1958.)
    >>> # Addition chains which do have this property are called "star chains."
    >>> # shortest_star_chain(2**6106 + 2**3048 + 2**2032 + 2**1016 + 1)
    >>> # This last example is somewhat shocking:
    >>> c191, l191 = shortest_star_chain(191)
    >>> c382, l382 = shortest_star_chain(382)
    >>> l191
    11
    >>> l382
    11
    """
    max_len = count_binary_digits(target, ones_factor=2) - 1
    record_state = isinstance(container, dict)

    logger.info("Length of doubling chain for N = %s: %s",
                target, max_len)
    logger.info("Pruning strategy: %s",
                'aggressive' if prune_aggressively else 'exhaustive')
    logger.info("Maintaining list of all star chains found: %s",
                "Yes" if isinstance(container, list) else "No")

    if record_state:
        logger.info("Requesting state history on: %s",
                    container.keys())

    start = time.time()
    cont_len, cont = _shortest_continuation_chain(
        target, seed, base_len=1, cont_len=0,
        max_cont_len=max_len-1, all_chains=container,
        prune_aggressively=prune_aggressively
    )
    end = time.time()

    if record_state and 'time' in container:
        pass

    logger.info("Algorithm execution time: %s s", end - start)
    return (cont_len + len(seed), seed + cont)

def _shortest_continuation_chain(target, previous, base_len=1,
                                 cont_len=None, max_cont_len=None,
                                 all_chains=None, prune_aggressively=True):

    def _report_found():
        if isinstance(all_chains, list):
            all_chains.append(previous)
        elif isinstance(all_chains, dict):
            all_chains['count'] = all_chains.get('count', 0) + 1

    if cont_len is None:
        cont_len = len(previous) - base_len

    last = previous[-1] if len(previous) else 0
    logger.debug("%sPartial chain: %s Last elem: %s Trying all continuations...",
                 INDENT*cont_len, previous, last)

    if last > target or (max_cont_len is not None and cont_len > max_cont_len):
        logger.debug("%sChain cannot be continued!", INDENT*cont_len)
        return (None, None)

    if last == target:
        _report_found()

        logger.debug("%sChain is already complete!", (INDENT*cont_len).replace(' ', '-'))
        logger.info("%sComplete chain found with length %s",
                    (INDENT*cont_len).replace(' ', '-'), base_len + cont_len)
        return (cont_len, [])

    best_len, best_chain = (None, None)
    for idx in range(base_len + cont_len):

        elem = previous[-idx-1]
        logger.debug("%sTrying %s as next diff, adding %s to chain...", INDENT*cont_len, elem, elem + last)

        opt_len, opt_chain = _shortest_continuation_chain(
            target, previous + [last + elem], base_len=base_len,
            cont_len=cont_len + 1, max_cont_len=max_cont_len,
            all_chains=all_chains, prune_aggressively=prune_aggressively
        )

        if opt_len is None:
            logger.debug("%sContinuation with %s failed, trying next element of chain %s", INDENT*cont_len, last+elem, previous)
            continue

        if best_len is None or opt_len < best_len:
            logger.debug("%sNew optimum length %s found!", INDENT*cont_len, opt_len)
            best_len, best_chain = opt_len, [last + elem] + opt_chain

            if prune_aggressively:
                max_cont_len = min(best_len, max_cont_len)

    logger.debug("%sPartial chain %s has optimal continuation %s",
                 INDENT*cont_len, previous, best_chain)
    return (best_len, best_chain)

def test_chain(c, l, n):
    logger.debug("Testing chain %s with length %s and target %s...", c, l, n)
    assert c is not None, \
        "I don't know why you passed me 'None' to verify, but honestly it's kind of a dick move"
    assert len(c) and c[0] == 1, \
        "Well-formed chains should be nonempty and should begin with the number 1"
    for j in range(1, l):
        assert (diff := c[j] - c[j-1] in c[:j]), \
            "Chain {chain} fails at index {idx}: diff = {diff} is not among the first {idx} elements!"\
            .format(idx=j, diff=diff, chain=c)
    assert c[-1] == n, \
        "Target {target} was not reached!  Final element of chain is {final}".format(target=n, final=c[-1])
    assert l == len(c), \
        "Chain has length {actual}, not {claimed}!".format(actual=len(c), claimed=l)

def count_binary_digits(num, zeros_factor=1, ones_factor=1):
    """
    >>> count_binary_digits(11)
    4
    >>> count_binary_digits(11, ones_factor=2)
    7
    >>> count_binary_digits(11, ones_factor=1, zeros_factor=2)
    5
    >>> count_binary_digits(15, ones_factor=2)
    8
    """
    count = 0
    while num:
        # if count_ones_as_extra:
            # count += num % 2
        digit = num % 2
        count += digit*ones_factor + (1^digit)*zeros_factor
        num >>= 1
    return count

def construct_doubling_sequence(target):
    """
    >>> construct_doubling_sequence(11)
    [1, 2, 4, 5, 10, 11]
    >>> construct_doubling_sequence(15)
    [1, 2, 3, 6, 7, 14, 15]
    """
    num_digits = count_binary_digits(target)
    ret = []
    for j in range(1, num_digits + 1):
        shift = num_digits - j
        ret.extend(sorted(
            {ret[-1] << 1 if len(ret) != 0 else 1, target >> shift}
        ))
    return ret

def main(target, benchmark=False):
    """
    >>> main(15)
    Doubling chain for N = 15 has length 7:
             [1, 2, 3, 6, 7, 14, 15]
    Optimal chain for N = 15 has length 6:
             [1, 2, 4, 5, 10, 15]
    >>> # This last example is somewhat shocking:
    >>> c191, l191 = shortest_star_chain(191)
    >>> main(191)
    >>> c382, l382 = shortest_star_chain(382)
    >>> l191
    11
    >>> l382
    11
    """
    start = time.time()
    length, chain = shortest_star_chain(target)
    algo_time = time.time() - start

    start = time.time()
    doubling_chain = construct_doubling_sequence(target)
    doubling_time = time.time() - start

    start = time.time()
    test_chain(chain, length, target)
    testing_time = time.time() - start

    if benchmark:
        print(
            """
Execution time: \n\t{algo_time} s to find optimal chain\n\t{d_time} s to find doubling chain\n\t{test_time} s to test the results
            """.format(algo_time=algo_time, d_time=doubling_time, test_time=testing_time)
        )

    print(
        "Doubling chain for N = {N} has length {L}:".format(
            N=target, L=len(doubling_chain))
    )
    print("\t", doubling_chain)
    print(
        "Optimal chain for N = {N} has length {L}:".format(
            L=length, N=target)
    )
    print("\t", chain)

if __name__ == "__main__":
    import sys
    main(sys.argv[1])
