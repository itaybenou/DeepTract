from utils.train_utils import *
from utils.Network import *
from utils.data_handling import *
from keras.losses import categorical_crossentropy
from keras.optimizers import *
from keras.metrics import categorical_accuracy, top_k_categorical_accuracy
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.model_selection import train_test_split
from os.path import join
import logging


class Trainer(object):

    def __init__(self, logger=None, **args):
        """
        :param logger: Logger object.
        :param args: Dictionary for storing config file parameters.
        """
        super().__init__()
        self.params = args['params']
        if logger is None:
            logging.basicConfig(format='%(asctime)s %(message)s')
            self.logger = logging.getLogger(__name__)
            self.logger.setLevel(logging.DEBUG)
        else:
            self.logger = logger
        self.model = None
        self.layers_size = self.params['layers_size']
        self.output_size = 725
        self.use_dropout = self.params['use_dropout']
        self.dropout_prob = self.params['dropout_prob']
        self.learning_rate = self.params['learning_rate']
        self.batch_size = self.params['batch_size']
        self.epochs = self.params['epochs']
        self.optimizer = self.params['optimizer']
        self.early_stopping = self.params['early_stopping']
        self.decay_LR = self.params['decay_LR']
        self.split_ratio = self.params['train_val_ratio']

        self.model_weights_save_dir = self.params['model_weights_save_dir']
        self.model_name = self.params['model_name']
        self.weights_file = join(self.model_weights_save_dir, self.model_name + '.hdf5')
        self.data_handler = DataHandler(self.params, mode='train')
        self.save_checkpoints = self.params['save_checkpoints']

    def get_model(self, grad_directions):

        # Num of steps in each batch
        max_streamline_length = np.max(get_streamlines_lengths(self.data_handler.tractogram))
        time_steps = int(max_streamline_length)

        self.model = DeepTract_network(time_steps, grad_directions, self.layers_size,
                                       self.output_size, self.use_dropout, self.dropout_prob)
        return time_steps

    def train(self):

        # Set data
        data_handler = self.data_handler
        data_handler.dwi = data_handler.mask_dwi()
        data_handler.dwi = data_handler.resample_dwi()
        data_handler.dwi = data_handler.max_val * data_handler.mask_dwi()
        grad_directions = data_handler.dwi.shape[3]
        dwi_means = calc_mean_dwi(data_handler.dwi, data_handler.wm_mask)
        vector_labels = get_geometrical_labels(data_handler.tractogram)
        x_train, x_valid, y_train, y_valid = train_test_split(data_handler.tractogram, vector_labels,
                                                              test_size=1-self.split_ratio)

        # Set model
        time_steps = self.get_model(grad_directions)

        # Set optimizer
        optimizer = self.optimizer(lr=self.learning_rate)

        # Set loss function
        loss = categorical_crossentropy

        # Compile model
        self.model.compile(loss=loss, optimizer=optimizer, metrics=[categorical_accuracy, sequence_top_k_categorical_accuracy])

        # Set callbacks
        callbacks = []
        if self.early_stopping:
            callbacks.append(EarlyStopping(monitor='val_categorical_accuracy',
                             patience=self.params['early_stopping_patience'],
                             verbose=1,
                             min_delta=1e-5,
                             mode='max'))
        if self.decay_LR:
            callbacks.append(ReduceLROnPlateau(monitor='val_categorical_accuracy',
                             factor=self.params['decay_factor'],
                             patience=self.params['decay_LR_patience'],
                             verbose=1,
                             min_delta=1e-5,
                             mode='max'))
        if self.save_checkpoints:
            callbacks.append(ModelCheckpoint(monitor='val_categorical_accuracy',
                             filepath=self.weights_file,
                             save_best_only=True,
                             mode='max'))

        # Training process
        train_history = self.model.fit_generator(
                generator=train_generator(data_handler.dwi, x_train, y_train, time_steps, self.output_size,
                                          self.batch_size, dwi_means),
                steps_per_epoch=np.ceil(float(len(x_train)) / float(self.batch_size)),
                epochs=self.epochs,
                verbose=1,
                callbacks=callbacks,
                validation_data=valid_generator(data_handler.dwi, x_valid, y_valid, self.output_size, self.output_size,
                                                self.batch_size, dwi_means),
                validation_steps=np.ceil(float(len(x_valid)) / float(self.batch_size)))

        # Save model
        # final_weights_path = os.path.join(self.model_weights_save_dir, self.model_name + '.pt')
        # self.logger.info('Saving final weights: %s', final_weights_path)
        # torch.save(model.state_dict(), final_weights_path)

        return train_history


def sequence_top_k_categorical_accuracy(y_true, y_pred, k=5):
    original_shape = K.shape(y_true)
    y_true = K.reshape(y_true, (-1, K.shape(y_true)[-1]))
    y_pred = K.reshape(y_pred, (-1, K.shape(y_pred)[-1]))
    top_k = K.cast(K.in_top_k(y_pred, K.argmax(y_true, axis=-1), k), K.floatx())
    return K.reshape(top_k, original_shape[:-1])