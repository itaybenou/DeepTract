import numpy as np
from dipy.data import get_sphere
from dipy.core.sphere import Sphere, HemiSphere
from keras import backend as K
from dipy.core.geometry import sphere_distance
from dipy.reconst.shm import sph_harm_lookup, smooth_pinv
from utils.data_handling import *
import threading


def get_geometrical_labels(streamlines):
    d_labels = []
    for i in range(len(streamlines)):
        directions = streamlines[i][1:, :] - streamlines[i][0:-1, :]
        directions = directions / np.linalg.norm(directions, axis=1)[:, None]
        d_labels.append(directions)

    return d_labels


def smooth_labels(directions, num_outputs=725):
    smoothed_labels = np.zeros((directions.shape[0], directions.shape[1], num_outputs), np.float32)
    sphere_points = get_sphere('repulsion724')

    for i in range(directions.shape[0]):
        aux_bool = 0
        for j in range(directions.shape[1]):
            if not (directions[i, j, 0] == 0.0 and directions[i, j, 0] == 0.0 and directions[i, j, 0] == 0.0):
                idx = sphere_points.find_closest(directions[i, j, :])
                labels_odf = np.exp(-1 * sphere_distance(directions[i, j, :], np.asarray(
                    [sphere_points.x, sphere_points.y, sphere_points.z]).T, radius=1.0) / 0.1)
                labels_odf = labels_odf / np.sum(labels_odf)
                smoothed_labels[i, j, :-1] = labels_odf
                smoothed_labels[i, j, -1] = 0.0
            elif aux_bool == 0:
                smoothed_labels[i, j, -1] = 1.0
                aux_bool = 1

    return smoothed_labels


def pad_and_convert2dwi(dwi_vol, X_batch, y_batch, max_length, DW_means):
    """
    INPUT: data - list of len=batch size , with data[i].shape= time_steps(i) x #gradient_directions
           labels - list of len=batch size , with labels[i].shape= time_steps(i) x 3
           batch_size - scalar

    OUTPUT: X_batch_padded - tensor of size batch size x max_time_steps x #gradient_directions
            y_batch_padded - tensor of size batch size x max_time_steps x 3
    """
    X_batch_padded = np.zeros((int(len(X_batch)), int(max_length), int(dwi_vol.shape[-1])))
    y_batch_padded = np.zeros((int(len(y_batch)), int(max_length), 3))
    for i in range(len(X_batch)):
        X_batch_padded[i, :X_batch[i].shape[0], :] = eval_volume_at_3d_coordinates(dwi_vol, X_batch[i]) - DW_means
        y_batch_padded[i, :y_batch[i].shape[0], :] = y_batch[i]

    return X_batch_padded, y_batch_padded


def sequence_top_k_categorical_accuracy(y_true, y_pred, k=5):
    original_shape = K.shape(y_true)
    y_true = K.reshape(y_true, (-1, K.shape(y_true)[-1]))
    y_pred = K.reshape(y_pred, (-1, K.shape(y_pred)[-1]))
    top_k = K.cast(K.in_top_k(y_pred, K.argmax(y_true, axis=-1), k), K.floatx())

    return K.reshape(top_k, original_shape[:-1])


def save_model(model, model_path):
    model_json = model.to_json()
    with open(model_path, "w", encoding='utf8') as json_file:
        json_file.write(model_json)


class ThreadSafeIterator:

    def __init__(self, it):
        self.it = it
        self.lock = threading.Lock()

    def __iter__(self):
        return self

    def __next__(self):
        with self.lock:
            return self.it.__next__()


def threadsafe_generator(f):
    """
    A decorator that takes a generator function and makes it thread-safe.
    """

    def g(*args, **kwargs):
        return ThreadSafeIterator(f(*args, **kwargs))

    return g


@threadsafe_generator
def train_generator(dwi_vol, X_list, y_list, time_steps, num_outputs, batch_size, dw_means):
    while True:
        shuffle_indices = np.arange(len(X_list))
        shuffle_indices = np.random.permutation(shuffle_indices)

        for start in range(0, len(X_list), batch_size):
            X_batch = []
            y_batch = []

            end = min(start + batch_size, len(X_list))
            idx_list = list(shuffle_indices[start:end])

            for idx in idx_list:
                X_batch.append(X_list[idx])
                # Augmentation: add the reversed streamline as well
                X_batch.append(np.flip(X_list[idx], axis=0))

                y_batch.append(y_list[idx])
                # Add the label of the reversed streamline
                y_batch.append(np.flip(-y_list[idx], axis=0))

            X_batch_padded, y_batch_padded = pad_and_convert2dwi(dwi_vol, X_batch, y_batch, time_steps, dw_means)
            y_batch_padded_smooth = smooth_labels(y_batch_padded, num_outputs)

            yield X_batch_padded, y_batch_padded_smooth


@threadsafe_generator
def valid_generator(dwi_vol, X_list, y_list, time_steps, num_outputs, batch_size, dw_means):
    while True:
        shuffle_indices = np.arange(len(X_list))
        shuffle_indices = np.random.permutation(shuffle_indices)

        for start in range(0, len(X_list), batch_size):
            X_batch = []
            y_batch = []

            end = min(start + batch_size, len(X_list))
            idx_list = list(shuffle_indices[start:end])

            for idx in idx_list:
                X_batch.append(X_list[idx])
                y_batch.append(y_list[idx])

            X_batch_padded, y_batch_padded = pad_and_convert2dwi(dwi_vol, X_batch, y_batch, time_steps, dw_means)
            y_batch_padded_smooth = smooth_labels(y_batch_padded, num_outputs)

            yield X_batch_padded, y_batch_padded_smooth
