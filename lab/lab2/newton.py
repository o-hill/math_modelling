'''

    Newton's Method solves f(x) = 0.

'''

import numpy as np
from ipdb import set_trace as debug
from scipy import optimize


def newtons_method(f, fp, x0):
    '''Solve f(x) = 0.

        @f: the function
        @fp: derivative of the function
        @x0: the initial guess
    '''

    tolerance = 1e-12
    miniter = 3

    x, i, convergence = x0, 0, False
    while not convergence:

        # Newton's update.
        x -= f(x) / fp(x)
        print(f'{i}: {x}, {abs(f(x))}')

        # Convergence criteria.
        convergence = i > miniter and abs(f(x)) < tolerance

        i += 1

    return x, i


def test_exp():
    '''Question 2(b) for the homework including plotting.'''

    plt.figure()

    f = lambda x: np.exp(x) - (4*x)
    fp = lambda x: np.exp(x) - 4

    x = np.linspace(0, 3, 2**10)

    newtons_method(f, fp, 0)

    acx = [0.35740296, 2.15329236]
    acy = [f(p) for p in acx]

    plt.subplot(1, 2, 1)
    plt.plot(x, f(x), label='e^x - 4x')
    plt.scatter(acx, acy)
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.grid()

    plt.subplot(1, 2, 2)
    plt.plot(x, [newtons_method(f, fp, x0)[0] for x0 in x])
    plt.xlabel('x_0')
    plt.ylabel('root, x_infinity')
    plt.grid()

    plt.show()


def test_cos():
    'tt''Question 2(c) for the homework including plotting.'''
    plt.figure()

    f = lambda x: np.cos(x ** 2) - x
    fp = lambda x: (-np.sin(x ** 2) * 2 * x) - 1

    actual_value = optimize.fsolve(f, 0)

    value, iterations = newtons_method(f, fp, 0)

    print(f'It took {iterations} iterations to converge to 12 digits.')
    print(f'Value found by fsolve: {actual_value}')

if __name__ == '__main__':

    from matplotlib import pyplot as plt

    # test_exp()
    test_cos()
















