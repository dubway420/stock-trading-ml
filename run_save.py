import numpy as np
from keras.models import load_model
from sklearn.externals import joblib
import matplotlib.pyplot as plt
from util import csv_to_dataset
from models import stacked_LSTM as tech_model
from keras import optimizers

stock = "IBM"

filenm = "data_daily/" + stock + "_daily.csv"

history_points = 50

offset = 2

ohlcv_histories, technical_indicators, next_day_open_values, unscaled_y, y_normaliser = csv_to_dataset(
    filenm,
    history_points,
    offset=1, next_day_only=True)
#
test_split = 0.9
n = int(ohlcv_histories.shape[0] * test_split)


unscaled_y_test = unscaled_y[n:]
# # real = plt.plot(unscaled_y_test, label='real')
# # plt.plot(y_normaliser.inverse_transform(technical_indicators[n:]))
# #
# # plt.show()
#
# #
#
# #
ohlcv_train = ohlcv_histories[:n]
tech_ind_train = technical_indicators[:n]
y_train = next_day_open_values[:n]
#
ohlcv_test = ohlcv_histories[n:]
tech_ind_test = technical_indicators[n:]
y_test = next_day_open_values[n:]
#
#
#
model = tech_model(history_points)
adam = optimizers.Adam(lr=0.0005)
model.compile(optimizer=adam, loss='mse')
trained_model = model.fit(x=[ohlcv_train, tech_ind_train], y=y_train, batch_size=32, epochs=100, shuffle=True,
                          validation_split=0.1)
#
history = trained_model.history

# evaluation

y_test_predicted = model.predict([ohlcv_test, tech_ind_test])
y_test_predicted = y_normaliser.inverse_transform(y_test_predicted)
y_predicted = model.predict([ohlcv_histories, technical_indicators])
y_predicted = y_normaliser.inverse_transform(y_predicted)
assert unscaled_y_test.shape == y_test_predicted.shape
real_mse = np.mean(np.square(unscaled_y_test - y_test_predicted))
scaled_mse = real_mse / (np.max(unscaled_y_test) - np.min(unscaled_y_test)) * 100
print(scaled_mse)

plt.gcf().set_size_inches(22, 15, forward=True)

start = 0
end = -1

real = plt.plot(unscaled_y_test[start:end], label='real')
pred = plt.plot(y_test_predicted[start:end], label='predicted')

# real = plt.plot(unscaled_y[start:end], label='real')
# pred = plt.plot(y_predicted[start:end], label='predicted')

plt.legend(['Real', 'Predicted'])

plt.show()

plt.plot(history['loss'], label="Loss")
plt.plot(history['val_loss'], label="Val Loss")

plt.plot([0, len(history['val_loss'])], [history['val_loss'][-1], history['val_loss'][-1]], ls='-', c='black')

plt.show()
#
# from datetime import datetime
#
file_name = "tech_model_" +stock + "_O" + str(offset) + ".H5"

model.save(file_name)



# result_type = 0

# model = load_model("tech_model_AAPL_O1.H5")
#
#
# y_test_predicted = y_normaliser.inverse_transform(model.predict([ohlcv_test, tech_ind_test]))
#
# # start = len(y_test_predicted) - 50
# start = 0
# end = 100
# # end = len(y_test_predicted)
#
# pred = plt.scatter(np.arange(start, end), y_test_predicted[start:end], label='predicted')
#
#
# real = plt.scatter(np.arange(start, end), unscaled_y_test[start:end], label='real')
#
# plt.legend()
# plt.show()
