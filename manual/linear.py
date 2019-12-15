import numpy as np
import matplotlib.pyplot as plt

n = 6
alp = 0.01
true_sig = [1., -1., 0., -.075, 4., -3.1]

C_log = []


def foo_gen(x):
    true_res = np.dot(x, true_sig)
    return true_res + 2. * np.random.random_sample() - 1.


def loss(y_hat1, y):
    return 1/len(y) * np.sqrt(np.sum((y_hat1 - y) ** 2))


# init
sig = np.random.rand(n)

x_data = np.random.random((100, 5)) * 20 - 10
x_data = np.hstack((np.ones((len(x_data[:, 0]), 1)), x_data))  # adding column with ones to original x data arr
y = foo_gen(x_data)

for i in range(200):
    y_hat = np.dot(x_data, sig)
    C = loss(y_hat, y)
    sig = sig - alp * 1/len(y) * np.dot((y_hat - y), x_data)

    C_log.append(C)


plt.plot(C_log)
plt.show()

print(C)
print(f"true sig: \n {true_sig}")
print(f"inferred sig: \n {sig}")
print(x_data) if C > 1. else print("fine")

