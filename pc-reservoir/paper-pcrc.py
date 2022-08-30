# CURRENTLY UNUSED

import numpy as np
from matplotlib import pyplot as plt

max_time = 10
# time-step
td = 0.2
# matrix number of elements
nx = 20
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
alpha_f = 0.5
# time constant
tau = 1
# membrane potential col vector
m = np.random.rand(nx, 1)
# neuron activity col vector
r = np.tanh(beta_m * m)


# helper method to initialize sparse connections in zeros matrix


def initialize_sparse(target, percentage, substitute):
    sparse_indices = np.random.choice(target.size, size=int(percentage * target.size), replace=False)
    np.put(target, sparse_indices, v=substitute)


initialize_sparse(w_rec, beta_rec, 1)
initialize_sparse(w_back, beta_back, 1)
initialize_sparse(w_e, beta_e, 1)


def input_gen(x):
    return 2 * x


# discretized model function
gen_model = input_gen(np.r_[0:max_time / td])

# calculate spectral radius of sparsely connected recurrent matrix
spectral_radius = max(abs(np.linalg.eigvals(w_rec)))
print("spectral radius", spectral_radius)

# incorporate strength of connections
w_rec = (alpha_rec * w_rec) / spectral_radius
w_back = alpha_back * w_back
w_e = alpha_e * w_e

error_hist = []

# initialize P(0) for calculation of Wout via FORCE
p = np.identity(nx) / alpha_f

# FORCE
for d in gen_model:
    y = np.maximum(0, w_out @ r)
    e = d - y

    v = p @ r
    p = p - ((v @ v.T) / (1 + v.T @ r))

    w_out = w_out - ((e @ v.T) / (1 + v.T @ r))

# FREEZE

for d in gen_model:
    y = np.maximum(0, w_out @ r)
    e = d - y

    v = p @ r
    p = p - ((v @ v.T) / (1 + v.T @ r))

    w_out = w_out - ((e @ v.T) / (1 + v.T @ r))

    m = m + 1 / tau * (-m + w_rec @ r + w_back @ y + alpha_e * w_e @ e)
    r = np.tanh(beta_m * m)

    error_hist += [e[0]]

plt.plot(error_hist)
plt.title('Error History')
plt.draw()
plt.show()
