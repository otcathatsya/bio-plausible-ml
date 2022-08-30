from time import time as tm

import numpy as np
import torch
import torchvision
from bindsnet.datasets import DataLoader, MNIST
from bindsnet.encoding import PoissonEncoder
from bindsnet.evaluation import all_activity, assign_labels
from bindsnet.models import DiehlAndCook2015
from bindsnet.network.monitors import Monitor
from torchvision.transforms import transforms

epochs = 1
batch_size = 32
neurons = 100
classes_mnist = 10
sim_time = 100
dt = 1.0

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('device!', device)

encoding = torchvision.transforms.Compose(
    [
        torchvision.transforms.ToTensor(),
        # for reasons beyond me this is incredibly important or network won't converge (intensity)
        transforms.Lambda(lambda x: x * 128)
    ]
)

train = MNIST(
    PoissonEncoder(time=sim_time, dt=dt),
    None,
    ".",
    download=True,
    transform=encoding,
)

# weird dataset wrapper but ok bindsnet
test = MNIST(
    PoissonEncoder(time=sim_time, dt=dt),
    None, '.',
    download=True,
    train=False,
    transform=encoding)

# will shuffle on every epoch (call to iter)
train_loader = DataLoader(
    train,
    batch_size=batch_size,
    shuffle=True
)

test_loader = DataLoader(
    test,
    batch_size=batch_size,
    shuffle=True
)

# take on dataset, iter will change len values for loaders
train_len = len(train)
test_len = len(test)
update_steps = int(train_len / batch_size / 10)
update_interval = update_steps * batch_size
print("Update interval of", update_interval)

# exc-inh values set equal according to (Diehl & Cook, 2015); norm and hyper-parameters taken from BindsNET defaults
network = DiehlAndCook2015(
    n_inpt=28 ** 2,
    n_neurons=100,
    exc=50,
    inh=50,
    dt=1.0,
    norm=78.4,
    nu=(1e-4, 1e-2),
    theta_plus=.05,
    inpt_shape=(1, 28, 28),
).to(device)

assignments = -torch.ones(neurons, device=device)
proportions = torch.zeros((neurons, classes_mnist), device=device)
rates = torch.zeros((neurons, classes_mnist), device=device)

accuracy_hist = []
spike_record = torch.zeros((update_interval, int(sim_time / dt), neurons), device=device)

spikes = {}
for layer in set(network.layers):
    spikes[layer] = Monitor(
        network.layers[layer], state_vars=["s"], time=int(sim_time / dt), device=device
    )
    network.add_monitor(spikes[layer], name="%s_spikes" % layer)

t1_train = tm()

for epoch in range(epochs):
    labels = []

    train_iter = iter(train_loader)
    for step in range(train_len):
        data = next(train_iter)
        X = data["encoded_image"].to(device)
        y = data["label"].to(device)
        # requires labels to be assigned so cannot be run first iteration
        if step % update_steps == 0 and step > 0:
            print("Epoch %s, step %s/%s" % (epoch, step * batch_size, train_len))

            label_tensor = torch.tensor(labels, device=device)

            pred = all_activity(
                spikes=spike_record, assignments=assignments, n_labels=classes_mnist
            )
            mean_acc = 100 * torch.sum(label_tensor.long() == pred).item() / len(label_tensor)
            accuracy_hist += [mean_acc]
            print("Current accuracy average: %.2f%%" % np.average(accuracy_hist))

            assignments, proportions, rates = assign_labels(
                spikes=spike_record,
                labels=label_tensor,
                n_labels=classes_mnist,
                rates=rates,
            )
            labels = []

        labels += y.tolist()

        # again, very strange convention for input dict
        network.run(inputs={"X": X}, time=sim_time, input_time_dim=1)

        # Add to spikes recording.
        s = spikes["Ae"].get("s").permute((1, 0, 2))
        spike_record[
        (step * batch_size)
        % update_interval: (step * batch_size % update_interval)
                           + s.size(0)
        ] = s

        network.reset_state_variables()

tTrain = tm() - t1_train
print("Training completed in %ss", tTrain)

accuracy_hist_test = []
# toggle off learning
network.train(mode=False)
t1_test = tm()

test_iter = iter(test_loader)
for step in range(test_len):
    data = next(test_iter)
    X = data["encoded_image"].to(device)
    y = data["label"].to(device)

    network.run(inputs={"X": X}, time=sim_time, input_time_dim=1)

    spike_record = spikes["Ae"].get("s").permute((1, 0, 2))

    label_tensor = torch.tensor(y, device=device)

    pred = all_activity(
        spikes=spike_record, assignments=assignments, n_labels=classes_mnist
    )

    mean_acc = 100 * torch.sum(label_tensor.long() == pred).item() / len(label_tensor)
    accuracy_hist_test += [mean_acc]

    network.reset_state_variables()

tTest = tm() - t1_test
print("Testing completed in %ss" % tTest)
print("Final test accuracy of %.2f%%" % np.average(accuracy_hist_test))
