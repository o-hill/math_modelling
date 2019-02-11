'''

    Question 1: Anonymous functions and iterative root-finding.

'''

import numpy as np
from matplotlib import pyplot as plt

from functools import reduce

from ipdb import set_trace as debug

x = [1]
g = lambda x: np.cos(x ** 2)

for _ in range(200):
    x.append(g(x[-1]))

xs = [p for p in x[:-1]]
ys = [p for p in x[1:]]

plt.subplot(1, 2, 1)
plt.plot(range(len(x)), x)
plt.title(f'Finding roots of g(x) for {len(x) - 1} iterations.')
plt.xlabel('Iteration')
plt.ylabel('g(x)')

plt.subplot(1, 2, 2)
plt.plot(xs, ys)
plt.show()

new = reduce(lambda res, func: func(res), [g for _ in range(200)], 1)
print(f'Original solution: {x[-1]}')
print(f'Reduce solution: {new}')
