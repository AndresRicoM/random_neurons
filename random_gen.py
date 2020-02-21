import numpy as np
import math

def random_x(exa, charac):

    new_data = np.random.rand(exa, charac)

    return new_data

def random_ydisc(exa, max):

    new_data = np.random.rand(exa, 1) * max
    new_data = np.round(new_data)

    return new_data

def random_ycont(exa):

    new_data = np.random.rand(exa, 1)

    return new_data

def complete_set(x, y):
    complete = np.hstack((x,y))
    return complete

x = random_x(20, 5)
y = random_ydisc(20, 10)
print(complete_set(x, y))
