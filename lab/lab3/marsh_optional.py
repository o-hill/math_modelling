
'''

    Solve the marsh linear programming problem. Question 5.

'''

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy import optimize

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
ex_machines = 2000                   # Cost of extra machines $/week
ex_labor = 500                       # Cost of extra labor $/week
investment = 500000                     # Total extra money for the season


# Use pandas dataframes for the dataset.
data = pd.read_excel('crops_data.xlsx')
n_constraints = 5
constraint_labels = np.array(['labor', 'land', 'machinery', 'irrigation', 'fertilizer'])
n_crops = len(data.index)

# Constraints.
b = np.array([
    max_labor,
    max_land,
    max_machine,
    max_irrigation,
    max_fertilizer,
    investment
])

A = np.array([
    np.array(data.iloc[:, 2].tolist() + [-1, 0]),                          # labor
    np.array(np.ones(n_crops).tolist() + [0, 0]),                           # land
    np.array(data.iloc[:, 3].tolist() + [0, -1]),                          # machinery
    np.array(np.array(data.iloc[:, 1] + 1).tolist() + [0, 0]) * grass,    # irrigation
    np.array(data.iloc[:, 4].tolist() + [0, 0]),                          # fertilizer
    np.array(np.zeros(n_crops).tolist() + [ex_labor, ex_machines])        # extra labor and machinery from investment
])

# Find the profit from each acre of each crop planted
# by subtracting all the costs from the yield.
rev = data.iloc[:, 6]
p_costs = data.iloc[:, 5]
f_costs = data.iloc[:, 4] * fertilizer_cost
m_costs = data.iloc[:, 3] * machine_cost * weeks_in_season
l_costs = data.iloc[:, 2] * labor_cost * weeks_in_season
profit = np.array(rev - p_costs - f_costs - m_costs - l_costs)
profit = np.array(profit.tolist() + [-ex_labor, -ex_machines])

# linprog minimizes, so take the negative.
f = -profit

bounds = [(0, None) for _ in range(len(f))]

result = optimize.linprog(f, A_ub=A, b_ub=b, bounds=bounds,
        options = { 'disp': False, 'tol': 1.8e-11 })

dual_f = b
dual_b = f
dual_A = -1 * A.T
bounds = [(0, None) for _ in range(len(dual_f))]
dual_res = optimize.linprog(dual_f, dual_A, dual_b, bounds=bounds,
        options={'disp': False, 'tol': 2.0e-10 })

lambdas = dual_res.x
profit = -result.fun

# Don't report tiny crops.
tolerance = 0.01
print('Optimal Crops:')
for index in np.where(result.x > tolerance)[0]:
    if index < n_crops:
        print(f'{data.iloc[index, 0]} {result.x[index]} acres')
    else:
        print(f'Extra machines and labor: {result.x[-1]}, {result.x[-2]}')

c = b - A.dot(result.x)
print(f'\nprofit: ${profit}')
print(f'Binding constraints:\n{constraint_labels[np.where(c < 1e-3)]}')
print(lambdas)















