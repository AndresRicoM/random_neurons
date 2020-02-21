import numpy as np
import math

def random_x(exa, charac):

    new_data = np.random.rand(exa, charac)

    return new_data

def random_ydisc(exa, range):

    new_data = np.random.rand(exa, 1)

    for rows in range(3): #new_data.shape[0]):
        print(rows)
        #new_data[rows,0] = new_data[rows,0].round

    return new_data

def random_ycont(exa):

    new_data = np.random.rand(exa, 1)

    return new_data

print(random_ydisc(3, 5))
