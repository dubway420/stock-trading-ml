import numpy as np
from keras.models import load_model
from sklearn.externals import joblib
import matplotlib.pyplot as plt
from util import csv_to_dataset
from models import stacked_LSTM as tech_model
from keras import optimizers
# from sklearn.metrics

history_points = 50

offset = 2

ohlcv_histories, technical_indicators, next_day_open_values, unscaled_y, y_normaliser = csv_to_dataset(
    'data_daily/AAPL_daily.csv',
    history_points,
    offset=offset, next_day_only=True)
#
test_split = 0.9
n = int(ohlcv_histories.shape[0] * test_split)

unscaled_y_test = unscaled_y[n:]

ohlcv_test = ohlcv_histories[n:]
tech_ind_test = technical_indicators[n:]
y_test = next_day_open_values[n:]

model = load_model("tech_model_IBM_O2.H5")

y_test_predicted = y_normaliser.inverse_transform(model.predict([ohlcv_test, tech_ind_test]))

# start = len(y_test_predicted) - 50
start = 0
end = 100
# end = len(y_test_predicted)



deltas_i = []
deltas_j = []

correct_delta = []

markers = []

x = unscaled_y_test[start]
y = y_test_predicted[start]

for i, j in zip(unscaled_y_test[start+1:end], y_test_predicted[start+1:end]):

    delta_i = 1 if i >= x else 0
    delta_j = 1 if j >= y else 0

    deltas_i.append(delta_i)
    deltas_j.append(delta_j)

    if delta_i == delta_j:
        correct_delta.append(1)
        markers.append('+')
    else:
        correct_delta.append(0)
        markers.append('x')

    x = i
    y = j

print(deltas_i)
print(deltas_j)


percent_correct = (sum(correct_delta) / len(correct_delta)) * 100
print("percent correct: " + str(percent_correct))

real = plt.scatter(np.arange(start+1, end), unscaled_y_test[start+1:end], label='real')
pred = plt.scatter(np.arange(start+1, end), y_test_predicted[start+1:end], label='predicted', c=correct_delta)

plt.legend()
plt.show()