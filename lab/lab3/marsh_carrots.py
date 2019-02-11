'''

    Investigate some questions for Question 3.

'''

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy import optimize

from tqdm import tqdm
from ipdb import set_trace as debug


grass = 3.3                             # acre-feet per acre - grass standard.
max_labor = 2500                        # person-hours/week - total available labor.
max_land = 7200                         # acre - total available agricultural land.
max_machine = 12000                     # Total available machine-hours/week.
rainfall = 18.5                         # Rainfall (acre-ft/acre)
max_irrigation = rainfall * max_land    # Irrigation (acre-ft)
max_fertilizer = 5000                   # Fertilizer (acre-ft)
weeks = 28                              # Weeks in a season
labor_cost = 10                         # Cost of labor/hour
machine_cost = 15                       # Cost of machinery/hour
fertilizer_cost = 5                     # Cost of fertilizer/acre-ft
weeks_in_season = 28


# Use pandas dataframes for the dataset.
data = pd.read_excel('crops_data.xlsx')
n_constraints = 5
constraint_labels = np.array(['labor', 'land', 'machinery', 'irrigation', 'fertilizer'])
n_crops = len(data.index)
carrots = np.where(data.iloc[:, 0] == 'carrots')[0][0]

# Constraints.
b = np.array([
    max_labor,
    max_land,
    max_machine,
    max_irrigation,
    max_fertilizer
])

A = np.array([
    np.array(data.iloc[:, 2]),                  # labor
    np.ones(n_crops),                           # land
    np.array(data.iloc[:, 3]),                  # machinery
    (np.array(data.iloc[:, 1]) + 1) * grass,    # irrigation
    np.array(data.iloc[:, 4])                   # fertilizer
])

bounds = [(0, None) for _ in range(n_crops)]

profits = { }
xopt = { }

# Now let's vary carrot yield.
orig_price = data.iloc[carrots, 6]
carrot_prices = np.linspace(orig_price, orig_price + 2000, 1000)
for i, carrot_price in tqdm(enumerate(carrot_prices)):

    # Find the profit from each acre of each crop planted
    # by subtracting all the costs from the yield.
    rev = np.array(data.iloc[:, 6])
    rev[carrots] = carrot_price
    p_costs = data.iloc[:, 5]
    f_costs = data.iloc[:, 4] * fertilizer_cost
    m_costs = data.iloc[:, 3] * machine_cost * weeks_in_season
    l_costs = data.iloc[:, 2] * labor_cost * weeks_in_season
    profit = np.array(rev - p_costs - f_costs - m_costs - l_costs)

    # linprog minimizes, so take the negative.
    f = -profit

    result = optimize.linprog(f, A_ub=A, b_ub=b, bounds=bounds,
                options = { 'disp': False })

    profits[i] = -result.fun
    xopt[i] = result.x

plt.figure()
plt.subplot(2, 1, 1)
plt.plot(carrot_prices, [p/100000 for p in profits.values()])
plt.xlabel('Price of Carrots ($/acre)')
plt.ylabel('Total Yield ($100000)')
plt.title('Price of Carrots vs. Total Yield')
plt.grid()


plt.subplot(2, 1, 2)
plt.plot(carrot_prices, [xopt[i][carrots] for i in range(len(xopt))])
plt.xlabel('Price of Carrots ($/acre)')
plt.ylabel('Amount of Carrots Planted (acres)')
plt.title('Price of Carrots vs. Total Amount of Carrots Planted')
plt.grid()

plt.subplots_adjust(hspace=0.5)

plt.show()








