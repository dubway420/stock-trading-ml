from util import csv_to_dataset_days
from keras.models import load_model
import matplotlib.pyplot as plt
import numpy as np



stock = "GOOGL"

filenm = "data_daily/" + stock + "_daily.csv"

history_points = 50

days = 5

ohlcv_histories, technical_indicators, next_day_open_values, unscaled_y, y_normaliser = csv_to_dataset_days(
    filenm,
    history_points,
    days)
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

model_name = "tech_model_" + stock + "_D5.H5"
model = load_model(model_name)

y_hat = model.predict([ohlcv_test, tech_ind_test])

for sample in range(0, len(y_hat), 5):

    plt.scatter(np.arange(len(y_test[sample])), y_test[sample], label="GT")
    plt.plot(np.unique(np.arange(len(y_test[sample]))), np.poly1d(np.polyfit(np.arange(len(y_test[sample])), y_test[sample], 2))(np.unique(np.arange(len(y_test[sample])))))

    plt.scatter(np.arange(len(y_hat[sample])), y_hat[sample], label="Pred")
    plt.plot(np.unique(np.arange(len(y_hat[sample]))), np.poly1d(np.polyfit(np.arange(len(y_hat[sample])), y_hat[sample], 2))(np.unique(np.arange(len(y_hat[sample])))))

    plt.legend()
    plt.show()

