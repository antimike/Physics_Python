# coding: utf-8
def print_binary(n):
    print('{0:b}'.format(n))
    
def print_binary(n):
    print('{0:b}'.format(n))
    
def binary_islands(x):
    x<<=1
    count = 0
    while x:
        s = x>>1
        if s%2 != x%2:
            count += 1
        x = s
    return count>>1
    
def binary_islands_popcount(x):
    return popcount(x^(x<<1)) >> 1
    
