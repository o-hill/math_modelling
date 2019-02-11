'''

    Newton's Method implemented with Automatic Differentiation.

'''

import autograd.numpy as np
from autograd import grad
from scipy import optimize

from tqdm import tqdm


def newton_ad(func, x0):
    '''Find the roots of the function using AD.'''

    tolerance = 1e-12
    miniter = 3

    x, i, convergence = float(x0), 0, False
    dfunc = grad(func)
    while not convergence:

        # Newton's update.
        x -= func(x) / dfunc(x)
        # print(f'{i}: {x}, {abs(func(x))}')

        # Convergence criteria.
        convergence = i > miniter and abs(func(x)) < tolerance
        i += 1

    return x

def f(x):
   return np.exp(-np.sqrt(x)) * np.sin(x * np.log(1 + (x ** 2)))

def g(x):
    return np.exp(-1 * np.sqrt(x))


if __name__ == '__main__':

    # Part 1
    h = lambda x: 1 + x**2
    h_ = grad(h)
    print(f"h(2): {h(2.0)}, h'(2) = {h_(2.0)}")

    solutions = set()
    for x in tqdm(np.linspace(1.0, 30.0, 50)):
        x = float(x)
        solutions.add(newton_ad(f, x))

    for root in solutions:
        print(f'Root: {root}, {f(root)}')
