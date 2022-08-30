import numpy as np
import pandas as pd
from keras import Sequential
from keras.callbacks import EarlyStopping
from keras.layers import Dense, LSTM, Conv1D, Flatten
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from tensorflow import keras


def plot_hist_regression(hist, y, fig_count):
    n_ = len(hist.history['mae'])
    axis[fig_count].plot(range(1, n_ + 1), np.asarray(hist.history['mae']), 'bo', label='MAE on training set')
    axis[fig_count].plot(range(1, n_ + 1), np.asarray(hist.history['val_mae']), 'b', label='MAE on validation set')
    axis[fig_count].legend()
    axis[fig_count].set_xlabel("Epoch")
    axis[fig_count].set_ylabel("MAE (degrees)")
    axis[fig_count].axhline(y=y)


df_train = pd.read_csv('DailyDelhiClimateTrain.csv')
df_test = pd.read_csv('DailyDelhiClimateTest.csv')

figure, axis = plt.subplots(2, 2)

axis[0, 0].plot(range(len(df_train)), df_train['meantemp'])
axis[0, 0].set_title("Mean Temperature")
axis[0, 0].set_ylabel("Temperature (Celsius)")

axis[0, 1].plot(range(len(df_train)), df_train['humidity'], color='orange')
axis[0, 1].set_title("Humidity")
axis[0, 1].set_ylabel("Humidity (vapor/m^3 air)")

axis[1, 0].plot(range(len(df_train)), df_train['wind_speed'], color='purple')
axis[1, 0].set_title("Wind Speed")
axis[1, 0].set_ylabel("Wind Speed (km/h)")

axis[1, 1].plot(range(len(df_train)), df_train['meanpressure'], color='green')
axis[1, 1].set_title("Mean Pressure")
axis[1, 1].set_ylabel("Mean Pressure (atm)")

figure.suptitle('Weather periodicity, entries 2013-2017')
plt.tight_layout()
plt.savefig('periodicity.pdf', bbox_inches='tight')
plt.show()

df_train = df_train.drop(['date'], axis=1)
df_test = df_test.drop(['date'], axis=1)

n_val_cutoff = int(len(df_train) * 0.8)

sampling_rate = 1
sequence_length = 14
delay = sampling_rate * sequence_length
batch_size = 16

predictors = ['meantemp', 'humidity', 'wind_speed', 'meanpressure']
y_cols = ['meantemp']

scaler_x = StandardScaler()

x_train = df_train.loc[:, predictors].to_numpy()
y_train = df_train.loc[:, y_cols].to_numpy()

x_test = df_test.loc[:, predictors].to_numpy()
y_test = df_test.loc[:, y_cols].to_numpy()

x_train = scaler_x.fit_transform(x_train)
x_test = scaler_x.transform(x_test)

lstm_train = keras.utils.timeseries_dataset_from_array(
    x_train[:-delay],
    targets=y_train[delay:],
    sampling_rate=sampling_rate, sequence_length=sequence_length,
    batch_size=batch_size,
    start_index=0, end_index=n_val_cutoff
)

lstm_val = keras.utils.timeseries_dataset_from_array(
    x_train[:-delay],
    targets=y_train[delay:],
    sampling_rate=sampling_rate, sequence_length=sequence_length,
    batch_size=batch_size,
    start_index=n_val_cutoff
)

lstm_test = keras.utils.timeseries_dataset_from_array(
    x_test[:-delay],
    targets=y_test[delay:],
    sampling_rate=sampling_rate, sequence_length=sequence_length,
    batch_size=batch_size,
    start_index=0
)

for X, target in lstm_train:
    print("X shape:", X.shape)
    print("target shape:", target.shape)
    break

# save for SNN conversion
y_test_snn = np.concatenate([y for x, y in lstm_test], axis=0)
np.savez_compressed("y_test.npz", y_test_snn)

x_test_snn = np.concatenate([x for x, y in lstm_test], axis=0)
np.savez_compressed("x_test.npz", x_test_snn)


def naive_method(dataset):
    error = 0
    samples = 0
    count = 0
    for X, target in dataset:
        pred = scaler_x.inverse_transform(X)[:, -1, 0]
        error += np.sum(np.abs(pred - target))
        samples += X.shape[0]
        count += 1
    return error / samples / batch_size


print('MAE on train set = %.2f (degrees Celsius)' % naive_method(lstm_train))
print('MAE on validation set = %.2f (degrees Celsius)' % naive_method(lstm_val))
print('MAE on test set = %.2f (degrees Celsius)' % naive_method(lstm_test))

baseline = naive_method(lstm_val)

# CNN MODEL

model_cnn = Sequential()
model_cnn.add(Conv1D(8, 3, activation='relu', input_shape=(sequence_length, len(predictors))))
model_cnn.add(Conv1D(8, 3, activation='relu', input_shape=(sequence_length, len(predictors))))
model_cnn.add(Flatten())
model_cnn.add(Dense(1))

model_cnn.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])

model_cnn.summary()

callbacks_cnn = [EarlyStopping(monitor='val_mae', patience=2)]
history_cnn = model_cnn.fit(lstm_train, epochs=10,
                            validation_data=lstm_val
                            , callbacks=callbacks_cnn)

# save model weights for SNN
keras.models.save_model(model_cnn, "cnn.h5", save_format='h5')

figure, axis = plt.subplots(1, 2)
plot_hist_regression(history_cnn, baseline, 0)

# LSTM MODEL

model_bl = Sequential()
model_bl.add(LSTM(32, input_shape=(sequence_length, len(predictors))))
model_bl.add(Dense(1))
model_bl.summary()

model_bl.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])

callbacks_lstm = [EarlyStopping(monitor='val_mae', patience=2)]
history_bl = model_bl.fit(lstm_train, epochs=60,
                          validation_data=lstm_val
                          , callbacks=callbacks_lstm)

plot_hist_regression(history_bl, baseline, 1)
figure.suptitle('MAE Convergence Train-Val compared to Baseline')
plt.tight_layout()
plt.savefig('mae_over_time.pdf', bbox_inches='tight')
plt.show()

Y = np.concatenate([y for x, y in lstm_test], axis=0)

cnn_mae = model_cnn.evaluate(lstm_test)[1]
lstm_mae = model_bl.evaluate(lstm_test)[1]

print('CNN Test MAE = %.2f degrees' % cnn_mae)
print('LSTM Test MAE = %.2f degrees' % lstm_mae)

test_pred_lstm = model_bl.predict(lstm_test)
test_pred_cnn = model_cnn.predict(lstm_test)

plt.figure(figsize=(20, 10))
plt.plot(Y, color='black', linewidth=2.0, label='Delhi Mean Temperature')
plt.plot(test_pred_lstm, color='orange', label='LSTM Prediction: %.2f degree MAE' % lstm_mae)
plt.plot(test_pred_cnn, color='purple', label='CNN Prediction: %.2f degree MAE' % cnn_mae)
plt.title('Delhi Mean Temp Prediction', fontsize=22)
plt.xlabel('Time', fontsize=22)
plt.xticks(fontsize=22)
plt.yticks(fontsize=22)
plt.ylabel('Mean Temp', fontsize=20)
plt.legend(prop={'size': 22})
plt.savefig('cnn-lstm-vs.pdf', bbox_inches='tight')
plt.show()
