from keras.models import Model
from keras.layers import Dense, Dropout, LSTM, Input, Activation


def simple_model(history_points):
    lstm_input = Input(shape=(history_points, 5), name='lstm_input')
    x = LSTM(100, name='lstm_0')(lstm_input)
    x = Dropout(0.2, name='lstm_dropout_0')(x)
    x = Dense(64, name='dense_0')(x)
    x = Activation('sigmoid', name='sigmoid_0')(x)
    x = Dense(1, name='dense_1')(x)
    output = Activation('linear', name='linear_output')(x)

    return Model(inputs=lstm_input, outputs=output)
