from keras.models import load_model
import numpy as np
np.random.seed(4)
from tensorflow import set_random_seed
set_random_seed(4)
from util import csv_to_dataset
import matplotlib.pyplot as plt
from sklearn.externals import joblib

model = load_model("technical_model.h5")

history_points = 50
# dataset

ohlcv_histories, technical_indicators, next_day_open_values, unscaled_y, y_normaliser = csv_to_dataset(
    'data_daily/AAPL_daily.csv',
    history_points,
    0, False)

joblib.dump(y_normaliser, "yscale.sve")


# test_split = 0.9
# n = int(ohlcv_histories.shape[0] * test_split)
#
# ohlcv_train = ohlcv_histories[:n]
# tech_ind_train = technical_indicators[:n]
# y_train = next_day_open_values[:n]
#
# ohlcv_test = ohlcv_histories[n:]
# tech_ind_test = technical_indicators[n:]
# y_test = next_day_open_values[n:]
#
# unscaled_y_test = unscaled_y[n:]
#
# y_test_predicted = model.predict([ohlcv_test, tech_ind_test])
# y_test_predicted = y_normaliser.inverse_transform(y_test_predicted)
#
# plt.plot(unscaled_y_test[:, 0])
# plt.plot(y_test_predicted[:, 0])
#
# plt.show()
#
# np.save("test_y", unscaled_y_test)
# np.save("ohlcv_test", ohlcv_test)
# np.save("tech_ind_test", tech_ind_test)




