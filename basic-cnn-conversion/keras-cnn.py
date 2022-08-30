import numpy as np
from keras import layers
from keras.callbacks import EarlyStopping
from keras.utils import to_categorical
from keras_flops import get_flops
from matplotlib import pyplot as plt
from tensorflow import keras


def plot_hist_regression(hist):
    n_ = len(hist.history['accuracy'])
    plt.plot(range(1, n_ + 1), np.asarray(hist.history['accuracy']), 'bo', label='Accuracy on training set')
    plt.plot(range(1, n_ + 1), np.asarray(hist.history['val_accuracy']), 'b', label='Accuracy on validation set')
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.show()


data = keras.datasets.cifar10

(train_x, train_y), (test_x, test_y) = data.load_data()

train_y = train_y.reshape(-1)
test_y = test_y.reshape(-1)

test_y = to_categorical(test_y, 10)
train_y = to_categorical(train_y, 10)

test_x = (test_x / 255.0).astype(np.float32)
train_x = (train_x / 255.0).astype(np.float32)

if len(train_x.shape) == 3:
    train_x = train_x.reshape(list(train_x.shape) + [3])
    test_x = test_x.reshape(list(test_x.shape) + [3])

# save for CNN
np.savez_compressed("x_test.npz", test_x)
np.savez_compressed("y_test.npz", test_y)
np.savez_compressed("x_norm.npz", train_x[::10])

model = keras.Sequential()

model.add(layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same', strides=(1, 1)))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))
model.add(layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same', strides=(1, 1)))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

model.build(input_shape=(None,) + train_x.shape[1:])
model.summary()

optimizer = keras.optimizers.Adam()

model.compile(optimizer=optimizer,
              loss="categorical_crossentropy",
              metrics=['accuracy'])

flops = get_flops(model, 300)
print(f"FLOPS: {flops} ")
print(f"FLOPS: {flops}")

callbacks_cnn = [EarlyStopping(monitor='val_accuracy', patience=2)]

history = model.fit(train_x, train_y,
                    batch_size=64,
                    epochs=30,
                    validation_data=(test_x, test_y),
                    callbacks=callbacks_cnn)

score = model.evaluate(test_x, test_y, verbose=0)
print('Test accuracy:', score[1])

keras.models.save_model(model, "mnist-cnn.h5", save_format='h5')

plot_hist_regression(history)

prediction = model(test_x)[0, :]
prediction_class = np.argmax(prediction)
print("calling", prediction_class)
