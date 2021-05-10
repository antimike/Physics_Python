from functools import cached_property
from collections import OrderedDict
from collections.abc import Sequence, Mapping
from typing import TypeVar, Union, Optional


T = TypeVar("T")
K = TypeVar("K")
Coords = Sequence[T]
NamedCoords = Sequence[K]
Params = Mapping[K, Union[T, Coords]]


class Direction:
    """class Direction.
    Represents a unique direction in Euclidean space.

    Public properties:
    ==================
    :coords: Cartesian coordinates of the unit vector corresponding to the represented direction
    :heading: Ordered pair (colatitude, longitude) == (th, ph) corresponding to the represented direction
    """

    """ Private helper functions and class properties """

    _heading_to_coords = OrderedDict(
        {
            "x": lambda th, ph: sin(th) * cos(ph),
            "y": lambda th, ph: sin(th) * sin(ph),
            "z": lambda th, ph: cos(th),
        }
    )

    _coords_to_heading = OrderedDict(
        {
            "th": lambda x, y, z: arccos(z / sqrt(x ^ 2 + y ^ 2 + z ^ 2)),
            "ph": lambda x, y, z: _azimuth(x, y),
        }
    )

    def _set_heading(self, coords: Coords) -> None:
        if len(coords) == 2:
            self._heading = tuple(coords)
        elif len(coords) == 3:
            self._heading = tuple(
                map(
                    lambda fn: fn(*coords).simplify(),
                    Direction._coords_to_heading.values(),
                )
            )
        else:
            raise ValueError(
                "Cannot construct heading with coordinate vector of length {l}".format(
                    l=len(coords)
                )
            )

    """ Constructor """

    def __init__(self, **kwargs: Params) -> None:
        """class Direction: constructor

        :kwargs: Keywords specifying either the Cartesian coordinates of a vector parallel to the desired direction or the colatitude / longitude of the corresponding unit vector.

        >>> d1 = Direction(x=1, y=2, z=3); d1.coords
        (1/14*sqrt(14), 1/7*sqrt(14), 3/14*sqrt(14))
        >>> d1.heading
        (arccos(3/14*sqrt(14)), arctan(2))
        >>> d2 = Direction(th=pi/2, ph=pi/4); d2.heading
        (1/2*pi, 1/4*pi)
        >>> d2.coords
        (1/2*sqrt(2), 1/2*sqrt(2), 0)
        >>> d3 = Direction(coords=(1, -2, 3)); d3.coords
        (1/14*sqrt(14), -1/7*sqrt(14), 3/14*sqrt(14))
        >>> d3.heading
        (arccos(3/14*sqrt(14)), -arctan(2))
        >>> d4 = Direction(heading=(3*pi/4, 5*pi/4)); d4.heading
        (3/4*pi, 5/4*pi)
        >>> d4.coords
        (-1/2, -1/2, -1/2*sqrt(2))
        >>> d5 = Direction(th=0, ph=3*pi); d5.heading
        (0, 3*pi)
        """
        self._set_heading(
            _extract_labeled_tuple(
                kwargs,
                0,
                coords=Direction._heading_to_coords.keys(),
                heading=Direction._coords_to_heading.keys(),
            )
        )

    """ Public properties """

    @cached_property
    def coords(self):
        """Direction.coords.
        The Cartesian unit vector associated with self's direction
        """
        return tuple(
            v(*self._heading).simplify() for v in Direction._heading_to_coords.values()
        )

    @cached_property
    def heading(self):
        """Direction.heading.
        The colatitude and longitude (th, ph) corresponding to self's direction
        """
        return self._heading


def _azimuth(x, y):
    """_azimuth.
    Returns the azimuthal angle corresponding to the point (x, y)

    :param x: Abcissa
    :param y: Ordinate
    """
    if not x * y == 0:
        return pi / 2 * (1 - sgn(x)) + arctan(y / x)
    elif x == 0:
        return sgn(y) * pi / 2
    else:
        return pi / 2 * (1 - sgn(x))


def _extract_labeled_tuple(
    d: Params, default: T, **key_lists: Mapping[K, NamedCoords]
) -> Optional[Coords]:
    """_extract_labeled_tuple.
    Returns an ordered tupple taken from `d` whose elements correspond to the keys of one of the `key_lists`.

    :param d: Inputs to extract labeled tuple from
    :param default: Default value for keys not found in d
    :param key_lists: Named lists of keys, e.g., heading=['th', 'ph']
    """
    for name, ll in key_lists.items():
        if len(ll & d.keys()):
            return [d.get(k, default) for k in ll]
        elif (c := d.get(name)) is not None:
            return c
    return None


#if __name__ == "__main__":
    #import doctest

    #doctest.testmod()
