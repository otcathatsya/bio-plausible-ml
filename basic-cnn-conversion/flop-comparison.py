import numpy as np

# torch/0.npz -> Reset by subtraction
# torch/1.npz -> Reset to zero
from matplotlib import pyplot as plt

log_vars = np.load('log/gui/test/log_vars/0.npz')
# 10^6 for 1m ops
print("synap", np.sum(log_vars['synaptic_operations_b_t']) * (10 ** 6))
print("ANN", np.sum(log_vars['neuron_operations_b_t']) * (10 ** 6))

# copied over from SNN logs
resetSubs_synap = 215683565
reset0_synap = 159619366
ann_flops = 576503999

X = [reset0_synap, resetSubs_synap]

plt.bar(range(len(X)), X, width=0.8, bottom=None, align='center', data=X, color=['orange', 'darkorchid'],
        edgecolor='black')
plt.xticks(range(len(X)), ['Reset to Zero', 'Reset by subtraction'])
plt.ylabel("Synaptic operations")
plt.xlabel("Spike reset method")
plt.axhline(ann_flops, color='firebrick')
plt.text(0.35, 546503999, 'ANN Operations')
plt.savefig("foo.pdf", bbox_inches='tight')
plt.show()
