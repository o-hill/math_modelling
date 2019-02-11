'''

    Question 1 - solve the farming linear program.

'''

import numpy as np
from scipy import optimize
from matplotlib import pyplot as plt

from ipdb import set_trace as debug


n_prices = 200
oat_price_min = 240
oat_price_max = 260

oat_price_range = np.linspace(
        oat_price_min,
        oat_price_max,
        n_prices)

b = np.array([1000, 300, 625])
A = np.array([
    [3, 1, 1.5],
    [0.8, 0.2, 0.3],
    [1, 1, 1]
])

bounds = [(0, b[2]) for _ in range(3)]

profit = { }
xopt = { }
lamb = { }

for i, oat_price in enumerate(oat_price_range):
    f = -np.array([400, 200, oat_price])

    result = optimize.linprog(f, A, b,
            bounds = bounds,
            options = { 'disp': False })

    if result.success:
        profit[i] = -1 * result.fun
        xopt[i] = result.x

        # Solve for the shadow prices.
        dual_c = b
        dual_b = f
        dual_A = -1 * A.T
        dual_res = optimize.linprog(dual_c, dual_A, dual_b,
                bounds = bounds, options = { 'disp': False })

        # debug()
        lamb[i] = np.array(dual_res.x)

    else:
        print(f'Failure to converge.')
        print(result)


# Plotting.
plt.figure()
plt.subplot(3, 1, 1)
plt.plot(oat_price_range, [p/1000 for p in profit.values()])
plt.xlabel('Price of Oats ($/acre)')
plt.ylabel('Total Yield (1,000 $)')
plt.title('Price Of Oats vs. Total Yield')
plt.grid()

plt.subplot(3, 1, 2)
corn, wheat, oats = zip(*xopt.values())
plt.plot(oat_price_range, corn, label='Corn')
plt.plot(oat_price_range, wheat, label='Wheat')
plt.plot(oat_price_range, oats, label='Oats')
plt.xlabel('Price of Oats ($/acre)')
plt.ylabel('Crops Planted (acre)')
plt.legend(loc='lower left')
plt.title('Price of Oats vs. Crops Planted')
plt.grid()

plt.subplot(3, 1, 3)
debug()
water, labor, land = zip(*lamb.values())
plt.plot(oat_price_range, water, label='Water')
plt.plot(oat_price_range, labor, label='Labor')
plt.plot(oat_price_range, land, label='Land')
plt.legend(loc='lower left')
plt.title('Price of Oats vs. Value of Constraints')
plt.xlabel('Price of Oats ($/acre)')
plt.ylabel('Shadow Price ($/unit)')
plt.grid()

plt.subplots_adjust(hspace=0.5)

plt.show()









