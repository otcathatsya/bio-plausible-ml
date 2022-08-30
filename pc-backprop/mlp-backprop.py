# UNUSED; backprop from scratch
# file IO
import csv
# exp, sqrt
import math
# initial weight assignment and data shuffling
import random as rand

# used for visualization; manual verification
import matplotlib.pyplot as plt

# hyper-parameters
sig_lambda = 0.9
learning_rate = 0.1
momentum_rate = 0.5
epoch_count = 10
val_rmse_min_threshold = 0.005


# Activation function
def sigmoid(val):
    return 1 / (1 + math.exp(-sig_lambda * val))


# Potential future implementation
def tanh(val):
    return math.tanh(val)


class Neuron:
    def __init__(self, inputs, **kwargs):
        self.weights = []
        self.delta_weights = []
        self.activation_value = 0.0
        self.inputs = inputs
        self.gradient_val = 0.0
        self.output_value = 0.0

        if 'output_value' in kwargs:
            self.output_value = kwargs.get('output_value')

        if 'activation_value' in kwargs:
            self.activation_value = kwargs.get('activation_value')

        # Bias boolean to exclude nodes in feed-forward process
        if 'bias' in kwargs:
            self.bias = True
        else:
            self.bias = False

        c = 0
        # Allow custom_weights kwargs to optionally test pre-made solutions; not used in final submission
        custom_weights = kwargs.get('custom_weights') if kwargs.get('custom_weights') else []
        while len(self.weights) < len(self.inputs):
            if custom_weights:
                self.weights += [custom_weights[c]]
            else:
                self.weights += [rand.random()]
            self.delta_weights += [learning_rate]
            c += 1

    # Feed-forward operation, activation function applied to sum of the weight-av products
    def update_activation(self):
        activation_sum = 0
        for idx, neuron in enumerate(self.inputs):
            activation_sum += neuron.activation_value * self.weights[idx]
        self.activation_value = sigmoid(activation_sum)


# NN class to generalize operations; model assumes full interconnection between all neurons
class NeuralNetwork:
    def __init__(self, input_layer, hidden_layer, output_layer):
        self.input_layer = input_layer
        self.hidden_layer = hidden_layer
        self.output_layer = output_layer

    def update_input(self, col_idx, updated_val):
        self.input_layer[col_idx].activation_value = updated_val

    def update_output(self, col_idx, updated_val):
        self.output_layer[col_idx].output_value = updated_val

    # Invoke feed-forward for all non-bias neurons; since each neuron has a reference to the previous layer,
    # loop over the last 2 (hidden-output)
    def pass_forward(self):
        for layer in [self.hidden_layer, self.output_layer]:
            for neuron in layer:
                if not neuron.bias:
                    neuron.update_activation()

    # Backpropagation process
    def backprop(self):
        grad_weight_sum = 0.0
        # Output layer gradient value
        for o_neuron in self.output_layer:
            error = o_neuron.output_value - o_neuron.activation_value
            o_neuron.gradient_val = sig_lambda * o_neuron.activation_value * (1.0 - o_neuron.activation_value) * error

        # Hidden layer gradient value
        for idx, h_neuron in enumerate(self.hidden_layer):
            # Sum of weight * gradient products for each output neuron connected to given hidden node
            for o_neuron in self.output_layer:
                grad_weight_sum += o_neuron.gradient_val * o_neuron.weights[idx]
            h_neuron.gradient_val = sig_lambda * h_neuron.activation_value * (
                    1.0 - h_neuron.activation_value) * grad_weight_sum
            grad_weight_sum = 0

        # Hidden layer delta weights
        for o_neuron in self.output_layer:
            for idx, weight in enumerate(o_neuron.weights):
                h_neuron = self.hidden_layer[idx]
                o_neuron.delta_weights[idx] = learning_rate * o_neuron.gradient_val * h_neuron.activation_value \
                                              + momentum_rate * o_neuron.delta_weights[idx]
                o_neuron.weights[idx] = weight + o_neuron.delta_weights[idx]

        # Input layer delta weights
        for h_neuron in self.hidden_layer:
            for idx, weight in enumerate(h_neuron.weights):
                i_neuron = self.input_layer[idx]
                h_neuron.delta_weights[idx] = learning_rate * h_neuron.gradient_val * i_neuron.activation_value \
                                              + momentum_rate * h_neuron.delta_weights[idx]
                h_neuron.weights[idx] = weight + h_neuron.delta_weights[idx]

    def store_weights(self):
        weights_total = retrieve_weights(self.hidden_layer) + retrieve_weights(self.output_layer);
        weight_file = open("weights.txt", "w")
        for weight in weights_total:
            weight_file.write(str(weight) + "\n")
        weight_file.close()


# Helper function to get weights before storing
def retrieve_weights(layer):
    res_weights = []
    for neuron in layer:
        for weight in neuron.weights:
            res_weights += [weight]
    return res_weights


# Construct NN object
input_layer0 = Neuron([], activation_value=1.0, bias=True)
input_layer1 = Neuron([])
input_layer2 = Neuron([])

hidden_layer0 = Neuron([], activation_value=1.0, bias=True)
hidden_layer1 = Neuron([input_layer0, input_layer1, input_layer2])
hidden_layer2 = Neuron([input_layer0, input_layer1, input_layer2])
hidden_layer3 = Neuron([input_layer0, input_layer1, input_layer2])
hidden_layer4 = Neuron([input_layer0, input_layer1, input_layer2])

output_layer1 = Neuron([hidden_layer0, hidden_layer1, hidden_layer2, hidden_layer3, hidden_layer4])
output_layer2 = Neuron([hidden_layer0, hidden_layer1, hidden_layer2, hidden_layer3, hidden_layer4])

nn = NeuralNetwork([input_layer0, input_layer1, input_layer2]
                   , [hidden_layer0, hidden_layer1, hidden_layer2, hidden_layer3, hidden_layer4]
                   , [output_layer1, output_layer2])

headers = ["X Distance", "Y Distance", "Velocity Y", "Velocity X"]


def read_column(file_name, row_name):
    with open(file_name, "r") as f:
        res = []
        reader = csv.DictReader(f, skipinitialspace=True)
        for row in reader:
            res += [float(row[row_name])]
    return res


def read_csv(file_name):
    csv_data = {header: [] for header in headers}
    for header in headers:
        csv_data[header] = read_column(file_name, header)
    return csv_data


# Helper method to update target activation values (input layer) and
# output values (output layer) to prepare for feed-forward
def update_nn(data, idx):
    input1 = data[headers[0]][idx]
    input2 = data[headers[1]][idx]
    output1 = data[headers[2]][idx]

    nn.update_input(1, input1)
    nn.update_input(2, input2)
    nn.update_output(0, output1)


def shuffle_dict(data):
    zipped = list(zip(data[headers[0]], data[headers[1]], data[headers[2]], data[headers[3]]))
    rand.shuffle(zipped)
    a, b, c, d = zip(*zipped)
    data[headers[0]] = a
    data[headers[1]] = b
    data[headers[2]] = c
    data[headers[3]] = d


train_data = read_csv("train.csv")
test_data = read_csv("train.csv")
validation_data = read_csv("train.csv")


# Helper function to calculate 1-pass MSE over given dataset
def calc_mse(data):
    mse_sum_clc = 0
    # for every data row
    for iX in range(0, len(data[headers[0]])):
        update_nn(data, iX)
        nn.pass_forward()
        mse_sum_clc += (((nn.output_layer[0].output_value - nn.output_layer[0].activation_value) ** 2)
                        + ((nn.output_layer[1].output_value - nn.output_layer[1].activation_value) ** 2)) / 2
    return mse_sum_clc / len(data[headers[0]])


# Monitor avg decrease for visualization and manual verification of stopping criteria using plt
rmse_decrease = []
prev_rmse = 0
# Training process
for i in range(0, epoch_count):
    print("Iteration #", i + 1)
    for iX in range(0, len(train_data[headers[0]])):
        update_nn(train_data, iX)
        nn.pass_forward()
        nn.backprop()
    rmse = math.sqrt(calc_mse(validation_data))
    if i >= 1:
        rmse_decrease += [prev_rmse - rmse]
        rmse_total = 0

        for rmse_val in rmse_decrease[-5:]:
            rmse_total += rmse_val
        rmse_avg = rmse_total / len(rmse_decrease)
        print("rmse avg decrease", rmse_avg)

        if rmse_avg < val_rmse_min_threshold or (prev_rmse - rmse) < 0:
            print("Early stopping criteria triggered")
            break
    prev_rmse = rmse

    # shuffle used sets
    shuffle_dict(validation_data)
    shuffle_dict(train_data)

# plot average mse validation decrease
plt.plot(rmse_decrease)
plt.xlabel("Epochs past 2nd")
plt.ylabel("RMSE Decrease Average")
plt.show()

final_mse_sum = 0
# RMSE Calculation, testing process
for i in range(0, epoch_count):
    final_mse_sum += calc_mse(test_data)
    # shuffle
    shuffle_dict(test_data)

rmse_final = math.sqrt(final_mse_sum / epoch_count)
print("RMSE", rmse_final)
print("Saving weights...")
nn.store_weights()
