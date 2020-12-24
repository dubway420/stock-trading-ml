from keras.models import Model
from keras.layers import Dense, Dropout, LSTM, Input, Activation, concatenate


def simple_model(history_points):
    lstm_input = Input(shape=(history_points, 5), name='lstm_input')
    x = LSTM(100, name='lstm_0')(lstm_input)
    x = Dropout(0.2, name='lstm_dropout_0')(x)
    x = Dense(64, name='dense_0')(x)
    x = Activation('sigmoid', name='sigmoid_0')(x)
    x = Dense(1, name='dense_1')(x)
    output = Activation('linear', name='linear_output')(x)

    return Model(inputs=lstm_input, outputs=output)

def tech_model(history_points):
    lstm_input = Input(shape=(history_points, 5), name='lstm_input')
    dense_input = Input(shape=(1,), name='tech_input')

    # the first branch operates on the first input
    x = LSTM(128, name='lstm_0')(lstm_input)
    x = Dropout(0.2, name='lstm_dropout_0')(x)
    lstm_branch = Model(inputs=lstm_input, outputs=x)

    # the second branch opreates on the second input
    y = Dense(128, name='tech_dense_0')(dense_input)
    y = Activation("relu", name='tech_relu_0')(y)
    y = Dropout(0.2, name='tech_dropout_0')(y)
    technical_indicators_branch = Model(inputs=dense_input, outputs=y)

    # combine the output of the two branches
    combined = concatenate([lstm_branch.output, technical_indicators_branch.output], name='concatenate')

    z = Dense(64, activation="sigmoid", name='dense_pooling')(combined)
    z = Dense(1, activation="linear", name='dense_out')(z)

    # our model will accept the inputs of the two branches and
    # then output a single value
    return Model(inputs=[lstm_branch.input, technical_indicators_branch.input], outputs=z)


def tech_model2(history_points):
    lstm_input = Input(shape=(history_points, 5), name='lstm_input')
    dense_input = Input(shape=(1,), name='tech_input')

    # the first branch operates on the first input
    x = LSTM(50, name='lstm_0')(lstm_input)
    x = Dropout(0.2, name='lstm_dropout_0')(x)
    x = LSTM(50, name='lstm_1')(x)
    x = Dropout(0.2, name='lstm_dropout_1')(x)
    lstm_branch = Model(inputs=lstm_input, outputs=x)

    # the second branch opreates on the second input
    y = Dense(20, name='tech_dense_0')(dense_input)
    y = Activation("relu", name='tech_relu_0')(y)
    y = Dropout(0.2, name='tech_dropout_0')(y)
    technical_indicators_branch = Model(inputs=dense_input, outputs=y)

    # combine the output of the two branches
    combined = concatenate([lstm_branch.output, technical_indicators_branch.output], name='concatenate')

    z = Dense(64, activation="sigmoid", name='dense_pooling')(combined)
    z = Dense(1, activation="linear", name='dense_out')(z)

    # our model will accept the inputs of the two branches and
    # then output a single value
    return Model(inputs=[lstm_branch.input, technical_indicators_branch.input], outputs=z)

def stacked_LSTM(history_points):

    lstm_input = Input(shape=(history_points, 5), name='lstm_input')
    dense_input = Input(shape=(1,), name='tech_input')

    # the first branch operates on the first input
    x = LSTM(128, name='lstm_0', return_sequences=True)(lstm_input)
    x = Dropout(0.2, name='lstm_dropout_0')(x)
    x = LSTM(128, name='lstm_1')(x)
    x = Dropout(0.2, name='lstm_dropout_1')(x)
    lstm_branch = Model(inputs=lstm_input, outputs=x)

    # the second branch opreates on the second input
    y = Dense(20, name='tech_dense_0')(dense_input)
    y = Activation("relu", name='tech_relu_0')(y)
    y = Dropout(0.2, name='tech_dropout_0')(y)
    y = Dense(20, name='tech_dense_1')(y)
    y = Activation("relu", name='tech_relu_1')(y)
    y = Dropout(0.2, name='tech_dropout_1')(y)
    technical_indicators_branch = Model(inputs=dense_input, outputs=y)

    # combine the output of the two branches
    combined = concatenate([lstm_branch.output, technical_indicators_branch.output], name='concatenate')

    z = Dense(64, activation="sigmoid", name='dense_pooling')(combined)
    z = Dense(1, activation="linear", name='dense_out')(z)

    # our model will accept the inputs of the two branches and
    # then output a single value
    return Model(inputs=[lstm_branch.input, technical_indicators_branch.input], outputs=z)
