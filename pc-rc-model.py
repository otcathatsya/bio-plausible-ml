import math

import matplotlib.pyplot as plt
import numpy as np

# simulation time
max_time = 50
# time delta
td = 1
# matrix number of elements
nx = 300
ny = 1
# recurrent conn matrix
w_rec = np.zeros((nx, nx))
# feedback conn matrices
w_back = np.zeros((nx, ny))
w_e = np.zeros((nx, ny))
# readout connections, to be trained
w_out = np.random.rand(ny, nx)
# % of sparse connectivity for recurrent connections
beta_rec = 0.1
# % of sparse connectivity for feedback connections
beta_back = 0.1
# % of sparse connectivity for feedback (error) connections
beta_e = 0.1
# scaling parameter for neuron activity
beta_m = 0.5
# strength of recurrent connections
alpha_rec = 0.15
# strength of feedback connections
alpha_back = 0.4
# error driven mode (1/0)
alpha_e = 1
# scaling parameter for initialization of p
alpha_f = 1
# time constant
tau = 1


# helper method to initialize sparse connections in zeros matrix
def initialize_sparse(target, percentage, substitute):
    sparse_indices = np.random.choice(target.size, size=int(percentage * target.size), replace=False)
    np.put(target, sparse_indices, v=substitute)


def input_gen(x):
    return math.sin(x) + 5


initialize_sparse(w_rec, beta_rec, 1)
initialize_sparse(w_back, beta_back, 1)
initialize_sparse(w_e, beta_e, 1)

# calculate spectral radius of sparsely connected recurrent matrix
spectral_radius = np.max(abs(np.linalg.eigvals(w_rec)))
print("spectral radius", spectral_radius)

# incorporate strength of connections
w_rec = alpha_rec * w_rec / spectral_radius
w_back = alpha_back * w_back
w_e = alpha_e * w_e

# discretized model function
generative_model = []
for x in range(0, max_time):
    generative_model += [input_gen(x)]
generative_model = np.asarray(generative_model)

# membrane potential col vector
m = np.random.rand(nx, 1)
# neuron activity col vector
r = np.random.rand(nx, 1)
# prediction col vector
y = np.zeros((ny, 1))
# error col vector
e = np.random.rand(ny, 1)
# initialize P(0) for calculation of Wout via FORCE
p = np.identity(nx) / alpha_f

res = []

for n in range(0, max_time):
    print("n @", n)
    y = np.maximum(np.zeros((ny, 1)), w_out @ r)
    e = generative_model[n] - y

    v = p @ r
    p = p - ((v @ np.transpose(v)) / (1 + np.transpose(v) @ r))
    w_out = w_out - ((e @ np.transpose(v)) / (1 + np.transpose(v) @ r))

    m = m + (1 / tau) * (-m + w_rec @ r + w_back @ y
                         + alpha_e * (w_e @ e))
    r = np.tanh(beta_m * m)

    res += [y[0]]

plt.plot(res)
plt.plot(generative_model)
plt.title('Approximation')
plt.draw()
plt.show()