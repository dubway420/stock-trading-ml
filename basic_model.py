from sklearn.metrics import mean_squared_error as mse
from keras.models import Model
from keras.layers import Dense, Dropout, LSTM, Input, Activation
from keras import optimizers
import numpy as np

np.random.seed(4)
from tensorflow import set_random_seed

set_random_seed(4)
from util import csv_to_dataset


# dataset

def model_training(filepath, model, history_points=50,  offset=0):
    ohlcv_histories, _, next_day_open_values, unscaled_y, y_normaliser = csv_to_dataset(filepath, history_points,
                                                                                        offset=offset)

    test_split = 0.9
    n = int(ohlcv_histories.shape[0] * test_split)

    ohlcv_train = ohlcv_histories[:n]
    y_train = next_day_open_values[:n]

    ohlcv_test = ohlcv_histories[n:]
    y_test = next_day_open_values[n:]

    unscaled_y_test = unscaled_y[n:]

    # model architecture

    model = model(history_points)

    adam = optimizers.Adam(lr=0.0005)
    model.compile(optimizer=adam, loss='mse')
    trained_model = model.fit(x=ohlcv_train, y=y_train, batch_size=32, epochs=100, shuffle=True, validation_split=0.1)

    # evaluation

    y_test_predicted = model.predict(ohlcv_test)
    y_test_predicted = y_normaliser.inverse_transform(y_test_predicted)
    y_predicted = model.predict(ohlcv_histories)
    y_predicted = y_normaliser.inverse_transform(y_predicted)

    assert unscaled_y_test.shape == y_test_predicted.shape
    real_mse = np.mean(np.square(unscaled_y_test - y_test_predicted))
    scaled_mse = real_mse / (np.max(unscaled_y_test) - np.min(unscaled_y_test)) * 100
    print(scaled_mse)

    import matplotlib.pyplot as plt

    plt.gcf().set_size_inches(22, 15, forward=True)

    start = 0
    end = -1

    real = plt.plot(unscaled_y_test[start:end], label='real')
    pred = plt.plot(y_test_predicted[start:end], label='predicted')

    # real = plt.plot(unscaled_y[start:end], label='real')
    # pred = plt.plot(y_predicted[start:end], label='predicted')

    plt.title(filepath)

    plt.legend(['Real', 'Predicted'])

    plt.show()

    from datetime import datetime
    model.save(f'basic_model.h5')

    history = trained_model.history

    plt.title(filepath)

    plt.plot(history['loss'], label="Loss")
    plt.plot(history['val_loss'], label="Val Loss")

    plt.plot([0, len(history['val_loss'])], [history['val_loss'][-1], history['val_loss'][-1]], ls='-')

    plt.show()

    return mse(unscaled_y_test, y_test_predicted)
