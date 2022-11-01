import pickle
from pathlib import Path
from time import time as tm

import matplotlib.pyplot as plt
import torch
import torchvision
from bindsnet.analysis.plotting import plot_voltages
from bindsnet.datasets import DataLoader, MNIST
from bindsnet.encoding import PoissonEncoder
from bindsnet.network import Network
from bindsnet.network.monitors import Monitor
from bindsnet.network.nodes import Input
from bindsnet.network.topology import Connection
from torch import nn
from torchvision.transforms import transforms

from ResNodes import BasicAccumulateNode


# similar to diehl but only map spikes, not to output directly
def gather_mapping(loader, data_length):
    voltage_ims = None
    voltage_axes = None
    t1_train = tm()
    spike_mappings = {}
    loader_iter = iter(loader)

    print("starting spike gathering")
    for step in range(data_length):
        if step * batch_size >= data_length:
            break

        if step * batch_size % (batch_size * 100) == 0:
            print("%s/%s" % (step * batch_size, data_length))

        data = next(loader_iter)
        X = data["encoded_image"].to(device)
        y = data["label"].to(device)

        network.run(inputs={"inp": X}, time=sim_time, input_time_dim=1)
        voltage_ims, voltage_axes = plot_voltages(
            {layer: voltages[layer].get("v").view(sim_time, -1) for layer in voltages},
            ims=voltage_ims,
            axes=voltage_axes,
            n_neurons={"res": (0, neurons)}
        )
        plt.pause(1e-8)

        enc = (y, data["image"]) if showcase else y
        spike_mappings[spikes["res"].get("s").sum(0)] = enc

        network.reset_state_variables()

    tTrain = tm() - t1_train
    print("Collected spike mappings in %ss" % tTrain)
    return spike_mappings


epochs = 1
batch_size = 32
neurons = 100
classes_mnist = 10
sim_time = 100
dt = 1.0
acc_to_thresh = {}

# whether to store full image data for later comparison, off unless showcasing
showcase = False
# whether to ignore the presence of a mappings.pkl file
from_scratch = True
# scales the amount of data to load for quick tests on non-HP devices; leave at 1.0 for any proper experiment study
dataset_fraction = 1.0

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('device!', device)

encoding = torchvision.transforms.Compose(
    [
        torchvision.transforms.ToTensor(),
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
train_len = int(len(train) * dataset_fraction)
test_len = int(len(test) * dataset_fraction)

update_steps = int(train_len / batch_size / classes_mnist)
update_interval = update_steps * batch_size
print("Update interval of", update_interval)

network = Network(dt=dt)
accs = []

# 28x28
input_layer = Input(28 ** 2, shape=(1, 28, 28))
network.add_layer(input_layer, name="inp")

thresh_range = range(-60, -50, 1)

for thresh in thresh_range:
    print("Using threshold", thresh)
    reservoir_layer = BasicAccumulateNode(neurons, thresh=thresh)

    network.add_layer(reservoir_layer, name="res")

    # input -> reservoir
    inp_res_connection = Connection(source=input_layer, target=reservoir_layer,
                                    w=0.5 * torch.randn(input_layer.n, reservoir_layer.n))
    network.add_connection(inp_res_connection, source="inp", target="res")

    # recurrent reservoir -> reservoir
    rec_res_connection = Connection(source=reservoir_layer, target=reservoir_layer,
                                    w=0.5 * torch.randn(reservoir_layer.n, reservoir_layer.n))
    network.add_connection(rec_res_connection, source="res", target="res")

    network = network.to(device)

    spikes = {}
    for layer in set(network.layers):
        spikes[layer] = Monitor(
            network.layers[layer], state_vars=["s"], time=int(sim_time / dt), device=device
        )
        network.add_monitor(spikes[layer], name="%s_spikes" % layer)

    voltages = {"res": Monitor(network.layers["res"], ["v"], time=sim_time, device=device)}
    network.add_monitor(voltages["res"], name="res_voltages")

    # load or gather mappings
    stored_mappings = Path("mappings.pkl")

    if stored_mappings.is_file() and not from_scratch:
        print("Mappings stored, opening")
        with open('mappings.pkl', 'rb') as f:
            train_mappings, test_mappings = pickle.load(f)
    else:
        print("training anew")
        train_mappings = gather_mapping(train_loader, train_len)
        test_mappings = gather_mapping(test_loader, test_len)

    # sigmoid readout
    readout_net = nn.Sequential(
        nn.Linear(neurons, classes_mnist),
        nn.Sigmoid()
    ).to(device)

    optimizer = torch.optim.SGD(readout_net.parameters(), lr=.05, momentum=.8)
    loss_fun = nn.MSELoss()

    print("Training readout")
    for epoch in range(epochs):
        print("Epoch %s/%s" % (epoch, epochs))
        for spikes, result in train_mappings.items():
            label = result
            optimizer.zero_grad()
            outputs = readout_net(spikes.float())
            label_batch = torch.zeros(batch_size, 10).float().to(device)
            for i, l in enumerate(label):
                label_batch[i, l] = 1.0

            loss = loss_fun(outputs, label_batch)

            loss.backward()
            optimizer.step()

    correct, total = 0, 0
    for spikes, result in test_mappings.items():
        label = result
        # extract from pair
        img = None
        outputs = readout_net(spikes.float())
        predicted = torch.argmax(outputs, dim=1)

        if showcase:
            figure, axis = plt.subplots(4, 8)
            for i in range(batch_size):
                col, row = divmod(i, 8)

                img_cpu = img[:, i].view(28, 28).detach().clone().cpu().numpy()
                axis[col, row].axis('off')
                axis[col, row].imshow(img_cpu)
                y_pred = predicted[i].item()
                y = label[i].item()
                if y_pred == y:
                    title_col = 'green'
                else:
                    title_col = 'red'
                axis[col, row].set_title(f"{y_pred}", color=title_col, fontweight='bold')

            figure.suptitle('MNIST RC Predictions')
            plt.tight_layout()
            plt.show()
            showcase = False

        total += batch_size
        correct += torch.sum(predicted == label)

    print(correct, "accurate out of", total)
    acc = 100 * correct / total

    print("Test Accuracy %.2f %%" % acc)
    acc_to_thresh[thresh] = acc
    accs += [acc]

with open('thresh_mappings.pkl', 'wb') as f:
    pickle.dump(acc_to_thresh, f)

print(accs)

for (attempted_thresh, acc) in acc_to_thresh.items():
    print("%.2f %% with a threshold of %s-mV" % (acc, attempted_thresh))
