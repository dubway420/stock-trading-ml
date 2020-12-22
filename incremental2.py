import numpy as np
from keras.models import load_model
from sklearn.externals import joblib
import matplotlib.pyplot as plt

sample = 0

model = load_model("technical_model.h5")

unscaled_y_test = np.load("test_y.npy")

ys = np.zeros(unscaled_y_test.shape)

ohlcv_test = np.load("ohlcv_test.npy")
tech_ind_test = np.load("tech_ind_test.npy")

y_normaliser = joblib.load("yscale.sve")

#####################################################
xa = np.zeros([1, ohlcv_test.shape[1], ohlcv_test.shape[2]])
xb = np.zeros([1, tech_ind_test.shape[1]])

xa[0] = ohlcv_test[sample]
xb[0] = tech_ind_test[sample]

####################################################
for i in range(len(unscaled_y_test)):

    y_hat = model.predict([xa, xb])

    xa[0, 0:-1] = xa[0, 1:]

    xa[0, -1] = y_hat[0]

    ys[i] = y_hat[0]


ys = y_normaliser.inverse_transform(ys)

plt.plot(unscaled_y_test[:, 0])
plt.plot(ys[:, 0])
plt.show()

