'''

    Analog of optimal loads schedule for matlab.

'''

import numpy as np
from matplotlib import pyplot as plt

def optimalloadschedule(a, X):

    A = np.diag(a)
    A[:-1, -1] = -1
    A[-1, :-1] = 1

    b = np.array([-1, -1, -1, X])

    x = np.linalg.lstsq(A, b)

    cost_rate = sum([(a_i * (x_i**2)) + x_i for x_i, a_i in zip(x[0][:-1], a[:-1])])
    x_ret = x[0][:-1]
    lamb = x[0][-1]

    return x_ret, lamb, cost_rate


if __name__ == '__main__':
    a = [0.0625, 0.0125, 0.0250, 0]
    X = 952

    print(optimalloadschedule(a, X))

    loads = []
    indep_line = np.linspace(600, 1200, 1000)
    for X in indep_line:
        loads.append(optimalloadschedule(a, X))

    plt.figure()

    # Plot each assignemnt for the generators along with the total.
    plt.subplot(1, 3, 1)
    plt.plot(indep_line, [x[0][0] for x in loads], label='x1')
    plt.plot(indep_line, [x[0][1] for x in loads], label='x2')
    plt.plot(indep_line, [x[0][2] for x in loads], label='x3')
    plt.plot(indep_line, [sum(x[0]) for x in loads], label='total')

    plt.title('Optimal Load Per Generator')
    plt.legend(loc='upper left')
    plt.xlabel('Mega Watts (MW)')
    plt.ylabel('Mega Watts (MW)')
    plt.grid()

    # Plot the optimal cost.
    plt.subplot(1, 3, 2)
    plt.plot(indep_line, [x[2] for x in loads], label='Optimal Cost')

    plt.title('Optimal Cost vs. Mega Watts')
    plt.xlabel('Mega Watts (MW)')
    plt.ylabel('Optimal Cost ($)')
    plt.grid()

    # Plot the Lagrange multiplier.
    plt.subplot(1, 3, 3)
    plt.plot(indep_line, [x[1] for x in loads])

    plt.title('Lagrange Multiplier vs. Mega Watts')
    plt.xlabel('Mega Watts (MW)')
    plt.ylabel('Lagrange Multiplier')
    plt.grid()

    plt.show()



