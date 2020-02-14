import keras
from keras.models import Sequential
from keras.layers import TimeDistributed, Dense, GRU, Masking, Dropout


def DeepTract_network(N_time_steps, grad_directions, num_neurons, num_outputs, use_dropout, dropout_prob):
    model = Sequential()
    model.add(Masking(mask_value=0., input_shape=(N_time_steps, grad_directions)))
    for iiLyr in range(len(num_neurons)):
        model.add(GRU(num_neurons[iiLyr], return_sequences=True))
        if use_dropout:
            model.add(Dropout(dropout_prob))
    model.add(TimeDistributed(Dense(num_outputs, activation='softmax')))

    return model