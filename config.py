from keras.optimizers import *


class Parameters(object):

    def __init__(self):
        self.params = dict()

        """ Model Parameters """

        # architecture_params - (list) Set number of neurons in each layer.
        self.params['layers_size'] = [1000, 1000, 1000, 1000]

        # use_dropout - (bool) Use dropout if "True"
        self.params['use_dropout'] = True

        # dropout_prob - (float) Dropout deletion probability (applies when use_dropout=True).
        self.params['dropout_prob'] = 0.3

        # model_name - (string) The model's name (used when saving weights file).
        self.params['model_name'] = 'DeepTract'

        # model_weights_save_dir - (string) Path for saving the model's files after training is done.
        self.params['model_weights_save_dir'] = "./trained_model"

        """ Training Parameters """

        # learning_rate -(float) Initial learning rate in training phase.
        self.params['learning_rate'] = 1e-4

        # optimizer - (keras.optimizers) Optimizer to be used in training.
        self.params['optimizer'] = Adam

        # batch_size - (int) Data batch size for training.
        self.params['batch_size'] = 8

        # epochs - (int) Number of training epochs.
        self.params['epochs'] = 50

        # decay_LR - (bool) Whether to use learning rate decay.
        self.params['decay_LR'] = True

        # decay_LR_patience - (int) Number of training epochs to wait in case validation performance does not improve
        # before learning rate decay is applied.
        self.params['decay_LR_patience'] = 5

        # decay_factor - (float [0, 1]) In an LR decay step, the existing LR will be multiplied by this factor.
        self.params['decay_factor'] = 0.5

        # early_stopping - (bool) Whether to use early stopping.
        self.params['early_stopping'] = True

        # early_stopping - (int) Number of epochs to wait before training is terminated when validation performance
        # does not improve.
        self.params['early_stopping_patience'] = 5

        # save_checkpoints - (bool) Whether to save model checkpoints during training.
        self.params['save_checkpoints'] = True

        """ Data Parameters """

        # DWI_path - (string) Path to the input DWI directory (should include .nii, .bvecs and .bvals files).
        self.params['DWI_path'] = "./data/dwi"

        # voxel_size - (list) DWI's voxel dimensions (in mm).
        self.params['voxel_size'] = [2, 2, 2]

        # tractogram_path - (string) Path to a tractogram (.trk file) to be used as training labels.
        self.params['tractogram_path'] = "./data/labels/tractography.trk"

        # brain_mask_path - (string) Path to a binary brain mask file that will be applied to the input DWI volume.
        # Insert None if such mask is not available.
        self.params['DWI_mask_path'] = "./data/mask/brain_mask.nii.gz"

        # wm_mask_path - (string) Path to a binary white natter mask file that will be applied to the input DWI volume.
        # Insert None if such mask is not available.
        self.params['DWI_mask_path'] = "./data/mask/wm_mask.nii.gz"

        # train_val_ratio - (float [0, 1]) Training/Validation split ratio for training.
        self.params['train_val_ratio'] = 0.9

        """ Inference (test-phase tractography) Parameters """

        # model_load_path - Path to the trained model's files (should include a .json and .hdf5 files).
        self.params['model_load_path'] = "./trained_model"

        # test_DWI_path - (string) Path to the tractography input DWI (.nii file).
        self.params['test_DWI_path'] = "./data/dwi/DWI.nii.gz"

        # brain_mask_path - (string) Path to a binary brain mask file that will be applied to the input DWI volume.
        # Insert None if such mask is not available.
        self.params['brain_mask_path'] = "./data/mask/brain_mask.nii.gz"

        # wm_mask_path - (string) Path to a binary white natter mask file that will be applied to the input DWI volume.
        # Insert None if such mask is not available.
        self.params['wm_mask_path'] = "./data/mask/wm_mask.nii.gz"

        # tractography_type - (string) 'deterministic' or 'probabilistic'
        self.params['tractography_type'] = 'deterministic'

        # N_seeds - (int) Number of seed points for tractography.
        self.params['N_seeds'] = 400000

        # step_size - (float) Tractography step size (in voxels).
        self.params['step_size'] = 0.5

        # max_angle - (float) Maximum allowed streamline angle (in degrees).
        self.params['max_angle'] = 60

        # max_length - (float) Maximum allowed streamline length (in mm).
        self.params['max_angle'] = 200

        # min_length - (float) Maximum allowed streamline length (in mm).
        self.params['min_length'] = 20

        # entropy_params (list) - [a, b, c] constants used to define the entropy threshold (see paper for more
        # details).
        self.params['entropy_params'] = [3, 10, 4.5]
