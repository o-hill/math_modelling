'''

    Question 3 on lab 2.

'''

from matplotlib import pyplot as plt
import autograd.numpy as np
from scipy import optimize

from ipdb import set_trace as debug
from autograd import grad

p = lambda t: 0.65 - (0.01 * t)
w = lambda t: 200 + (5 * t)
C = lambda t: 0.45 * t
profit = lambda t: p(t) * w(t) - C(t)

p_ = lambda t: -0.01
w_ = lambda t: 5
C_ = lambda t: 0.45

dpdt = lambda t: p_(t) * w(t) + p(t) * w_(t) - C_(t)

time_to_sell = optimize.fsolve(dpdt, 0)
print(f'Question 2: the best time to sell is at {time_to_sell} days')

# Plot out the function.
# x = np.linspace(0, 20, 1000)
# plt.plot(x, profit(x))
# plt.grid()
# plt.show()

w = lambda t, gamma: 200 * np.exp(gamma * t)
w_ = lambda t, gamma: 200 * gamma * np.exp(gamma * t)
dpdt = lambda t, g: p_(t) * w(t, g) + p(t) * w_(t, g) - C_(t)
profit = lambda t, g: p(t) * w(t, g) - C(t)

x = np.linspace(0.020, 0.030, 1000)
optimal = [ ]
for gamma in x:
    optimal.append(optimize.fsolve(dpdt, 0.5, args=(gamma)))

# x = np.linspace(0, 50, 1000)
# optimal = [profit(t, 0.025) for t in x]

# plt.subplot(1, 3, 1)
plt.plot(x, optimal)
plt.ylabel('Days')
plt.xlabel(r'$\gamma$')
plt.title(r'Optimal Time to Sell vs $\gamma$')
plt.grid()
# plt.subplot(1, 3, 2)
# x = np.linspace(0, 50, 1000)
# plt.plot(x, dpdt(x, 0.028))
# plt.grid()
# plt.subplot(1, 3, 3)
# plt.plot(x, dpdt(x, 0.02995))
# plt.grid()
# plt.show()
print(f'Optimal time to sell at gamma = 0.025: {optimize.fsolve(dpdt, 0.5, args=(0.025))}')
print(f'Profit at optimal time to sell: {profit(19.47, 0.025)}')

# Compute sensitivity of optimal time to sell to gamma.
# Or, dp/dgamma * gamma/p, with gamma = 0.025, p = 139.395
dpdg = lambda t, g: -2 * (t - 65) * t * np.exp(t * g)

print(f'Sensitivity of optimal selling time to gamma: {dpdg(19.468, 0.025) * (0.025/139.395)}')
print(f'Profit at g = 0.025: {optimize.fsolve(dpdt, 0.5, args=(0.025))}')
print(f'Profit at g = 0.025*1.01: {optimize.fsolve(dpdt, 0.5, args=(0.025*1.01))}')

print('New modelling for part (5)')
w = lambda t: 200 + 6 * (0.96 ** t)
p = lambda t: -(((t-10)**2)/300) + 0.75
w_ = lambda t: -0.244932 * (0.96**t)
p_ = lambda t: (10 - t)/150
profit = lambda t: p(t) * w(t) - C(t)
dpdt = lambda t: p_(t) * w(t) + p(t) * w_(t) - C_(t)

print(f'New optimal time to sell: {optimize.fsolve(dpdt, 0.5)}')

plt.figure()
x = np.linspace(0, 100, 1000)
plt.subplot(1, 2, 1)
plt.plot(x, profit(x))
plt.xlim(0, 25)
plt.ylim(-10, 160)
plt.xlabel('Days')
plt.ylabel('Profit')
plt.title('f(x)')
plt.grid()

plt.subplot(1, 2, 2)
plt.plot(x, dpdt(x))
plt.xlim(0, 15)
plt.ylim(-10, 20)
plt.xlabel('Days')
plt.ylabel('dp/dt')
plt.title("f'(x)")
plt.grid()

plt.show()






