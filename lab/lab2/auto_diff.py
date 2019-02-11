import autograd.numpy as np
from autograd import grad
from matplotlib import pyplot as plt


def sig(z):
    return 1 / (1 + np.exp(-z))

x = np.linspace(-8, 8, 1000)

plt.figure()
plt.subplot(1, 2, 1)
plt.plot(x, sig(x))
plt.title('sigmoid')

dsig_auto = grad(sig)


plt.subplot(1, 2, 2)
plt.plot(x, [dsig_auto(point) for point in x])
plt.title('dsigmoid')
plt.show()
