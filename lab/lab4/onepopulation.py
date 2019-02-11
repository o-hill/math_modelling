'''

    Python analog to onepopulation.m for question 1.

'''

import numpy as np
from scipy import integrate
from matplotlib import pyplot as plt

from ipdb import set_trace as debug


def one_population(question: int = 1, betas: list = [2.5], x0s: list = [[0.95, 0.05, 0.0]]) -> None:
    '''Solve the Kermack-McKendrick model for disease spread.'''

    N = 100                         # Total population size.
    betas = np.array(betas) / N     # Infection rate, 1/beta = typical time between contacts.
    gamma = 1.0                     # Recovery rate, 1/gamma = typical time until recovery.
    tf = 10                         # Final time.
    ys, ts, rbs = [ ], [ ], [ ]

    for beta in betas:
        for x0 in x0s:

            # Initial condition.
            x0 = N * np.array(x0)
            R0 = (beta * x0[0]) / gamma
            t0 = 0
            dt = 1

            # Simulate.
            r = integrate.ode(kerken)
            r.set_initial_value(x0, t0).set_f_params({ 'beta': beta, 'gamma': gamma })

            t, y = [t0], [x0]
            while r.successful() and r.t < tf:
                r.integrate(r.t + dt)
                t.append(r.t)
                y.append(r.y)

            ys.append(y)
            ts.append(t)
            rbs.append((R0, beta, x0))

    if question == 1:
        plot_output_q1(ys[0], ts[0], rbs[0])

    if question == 2:
        plot_output_q2(ys, ts, rbs)



def kerken(t: int = 0, x: tuple = ( ), params: dict = { }) -> np.ndarray:
    '''RHS of the Kermack-McKendrick ODE model.'''

    S, I, R = x

    dxdt = np.array([
        -params['beta'] * S * I,
        params['beta'] * S * I - (params['gamma'] * I),
        params['gamma'] * I
    ])

    return dxdt


def plot_output_q1(y: list = [ ], t: list = [ ]) -> None:
    '''Plot the appropriate output for question 1.'''

    S, I, R = zip(*y)
    S, I, R = np.array(S), np.array(I), np.array(R)

    # Plotting...
    plt.figure()

    plt.subplot(2, 1, 1)
    label_list = ['Susceptible Population', 'Infected Population', 'Recovered Population']
    for var, label in zip([S, I, R], label_list):
        plt.plot(t, var, label=label)

    plt.legend(loc='best')
    plt.xlabel('Time')
    plt.ylabel('Compartments')
    plt.title(fr'$R_0$ = {R0}')
    plt.grid()

    plt.subplot(2, 1, 2)
    plt.plot(1/2 * (2*R+I) / N, np.sqrt(3)/2 * I / N)
    plt.plot([0, 0.5, 1, 0], [0, np.sqrt(3)/2, 0, 0])

    plt.show()


def plot_output_q2(ys: list = [ ], ts: list = [ ], rs: list = [ ]) -> None:
    '''Plot the output for question 2.'''

    plt.figure()
    lines = [ ]
    debug()
    labels = ['Susceptible Population', 'Infected Population', 'Recovered Population']
    titles = [fr'$\beta$ = {b[1]}' for i, b in enumerate(rs) if i % 3 == 0]
    cols = [fr'$X_0$ = {b[2]}' for b in rs]

    for i, tup in enumerate(zip(ys, ts, rs)):

        y, t, rb = tup
        S, I, R = zip(*y)
        S, I, R = np.array(S), np.array(I), np.array(R)

        plt.subplot(3, 3, i + 1)
        for var, label in zip([S, I, R], labels):
            line, = plt.plot(t, var)
            lines.append(line)

        if i < 3:
            plt.title(titles[i])

        if i % 3 == 0:
            plt.ylabel(cols[int(i/3)])

        # plt.xlabel('Time')
        # plt.ylabel('Compartments')
        # plt.title(fr'$R_0$ = {rb[0]}, $\beta$ = {rb[1]}')
        plt.grid()

    plt.subplots_adjust(hspace=0.5)
    plt.figlegend((lines[0], lines[1], lines[2]), labels)
    plt.show()



if __name__ == '__main__':

    # print('\n------------------------')
    # print('Question 1')
    # one_population()
    # print('------------------------\n')

    print('\n------------------------')
    print('Question 2')
    one_population(question = 2,
            betas = list(np.linspace(0, 2.5, 3))[::-1],
            x0s = [[0.95, 0.05, 0.0], [0.85, 0.15, 0.0], [0.75, 0.25, 0.0]])
    print('------------------------\n')




