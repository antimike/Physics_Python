from enum import Enum, auto

def find_key_recursive(key, structure):
    if key is in structure:
        return (structure, 1)
    elif hasattr(structure, key):
        return 


def Inclusion(Enum):
    ATTR = auto()
    KEY = auto()
    ELEM = 
