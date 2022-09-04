import pickle
from pathlib import Path
from time import time as tm

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from bindsnet.datasets import DataLoader, MNIST
from bindsnet.encoding import PoissonEncoder
from bindsnet.network import Network
from bindsnet.network.monitors import Monitor
from bindsnet.network.nodes import Input, LIFNodes
from bindsnet.network.topology import Connection
from torch import nn
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

network = Network(dt=dt)

input_layer = Input(784, shape=(1, 28, 28))
network.add_layer(input_layer, name="inp")

reservoir_layer = LIFNodes(neurons, thresh=-52 + np.random.randn(neurons).astype(float))
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


# similar to diehl but only map spikes, not to output directly
def gather_mapping(loader, data_length):
    t1_train = tm()
    spike_mappings = {}
    loader_iter = iter(loader)
    print("starting spike gathering")
    for step in range(data_length):
        if step * batch_size % (batch_size * 100) == 0:
            print("%s/%s" % (step * batch_size, len(train)))
        if step * batch_size >= data_length:
            break

        data = next(loader_iter)
        X = data["encoded_image"].to(device)
        y = data["label"].to(device)
        # requires labels to be assigned so cannot be run first iteration

        network.run(inputs={"inp": X}, time=sim_time, input_time_dim=1)
        spike_mappings[spikes["res"].get("s").sum(0)] = (y, data["image"])

        network.reset_state_variables()

    tTrain = tm() - t1_train
    print("Collected spike mappings in %ss" % tTrain)
    return spike_mappings


stored_mappings = Path("mappings.pkl")

if not stored_mappings.is_file():
    print("Not stored, gathering mappings")
    train_mappings = gather_mapping(train_loader, len(train))
    test_mappings = gather_mapping(test_loader, len(test))

    # save mappings for showcase use without training again
    with open('mappings.pkl', 'wb') as f:
        pickle.dump([train_mappings, test_mappings], f)
else:
    print("Mappings stored, opening")
    with open('mappings.pkl', 'rb') as f:
        train_mappings, test_mappings = pickle.load(f)

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
        label, _ = result
        optimizer.zero_grad()
        outputs = readout_net(spikes.float())
        label_batch = torch.zeros(batch_size, 10).float().to(device)
        for i, l in enumerate(label):
            label_batch[i, l] = 1.0

        loss = loss_fun(outputs, label_batch)

        loss.backward()
        optimizer.step()

correct, total = 0, 0
showcase = True
for spikes, result in test_mappings.items():
    label, img = result
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

print(
    "\n Accuracy of the model on %d test images: %.2f %%"
    % (len(test), 100 * correct / total)
)
