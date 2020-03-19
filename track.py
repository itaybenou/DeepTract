from utils.test_utils import *
from utils.data_handling import *
from os.path import join
from keras.models import model_from_json
import logging
from itertools import compress
from tqdm import tqdm
from nibabel import streamlines


class Tracker:

    """
    Class for running tractography using a trained DeepTract model.
    """

    def __init__(self, logger=None, **args):
        """
        :param logger: Logger object.
        :param args: Dictionary for storing config file parameters (as **kwargs).
        """
        super().__init__()
        self.params = args['params']
        if logger is None:
            logging.basicConfig(format='%(asctime)s %(message)s')
            self.logger = logging.getLogger(__name__)
            self.logger.setLevel(logging.DEBUG)
        else:
            self.logger = logger

        self.trained_model_dir = self.params['trained_model_dir']
        self.model = None
        self.load_model()
        self.data_handler = DataHandler(self.params, mode='track')
        self.save_tractogram = self.params['save_tractogram']
        self.save_dir = self.params['save_dir']

        self.dwi_means = None
        self.tractography_type = self.params['tractography_type']
        self.num_seeds = self.params['num_seeds']
        self.track_batch_size = self.params['track_batch_size']
        self.step_size = self.params['step_size']
        self.max_angle = self.params['max_angle']
        self.max_length = self.params['max_length']
        self.min_length = self.params['min_length']
        self.track_length = self.model.input_shape[1]
        self.entropy_params = self.params['entropy_params']
        self.entropy_th = calc_entropy_threshold(self.entropy_params, self.track_length)

    def load_model(self):
        """
        Loads a trained DeepTract model from .json and weights files.
        """
        json_file = get_file_path(os.getcwd(), self.trained_model_dir, "*.json*")
        weights_file = get_file_path(os.getcwd(), self.trained_model_dir, "*.h*")
        model_json = open(json_file, 'r')
        loaded_model_json = model_json.read()
        model_json.close()
        model = model_from_json(loaded_model_json)
        model.load_weights(weights_file)
        self.model = model
        return

    @staticmethod
    def get_seed_mask(data_handler):
        """
        Returns a binary mask (WM / brain mask, if exist) to draw seed points from
        """
        if data_handler.wm_mask.size > 0:
            return data_handler.wm_mask
        elif not data_handler.brain_mask.size > 0:
            return data_handler.brain_mask
        else:
            return np.ones_like(data_handler.dwi)

    def streamline_tracking(self, data_handler, seed_points, num_batches):
        """
        Performs iterative streamline tractography using the trained DeepTract model.
        """

        sphere724 = get_sphere('repulsion724')
        angles724 = calc_angles_matrix(sphere724)
        angles725 = np.hstack(
            (np.vstack((angles724, np.zeros(angles724.shape[1]))), np.zeros((angles724.shape[0] + 1, 1))))

        dilated_wm_mask = mask_dilate(data_handler.wm_mask)
        all_streamlines = []

        for batch_idx in tqdm(range(num_batches)):
            # Initialize dwi inputs
            start_idx = batch_idx*self.track_batch_size
            end_idx = min((batch_idx + 1)*self.track_batch_size, seed_points.shape[0])
            seeds_batch = zero_pad_seeds(seed_points[start_idx:end_idx, :], end_idx-start_idx, self.track_length)
            dwi_inputs = np.zeros((seeds_batch.shape[0], seeds_batch.shape[1], len(self.dwi_means)))
            dwi_inputs[:, 0, :] = eval_volume_at_3d_coordinates(data_handler.dwi, seeds_batch[:, 0, :]) - self.dwi_means
            batch_streamlines = [np.expand_dims(seeds_batch[i, 0, :], axis=0) for i in range(seeds_batch.shape[0])]

            # Initialize track termination masks
            EoF_mask = np.zeros(len(dwi_inputs), dtype=bool)
            entropy_mask = np.zeros(len(dwi_inputs), dtype=bool)
            angle_mask = np.zeros(len(dwi_inputs), dtype=bool)
            inWM_mask = np.ones(len(dwi_inputs), dtype=bool)

            next_positions = seeds_batch[:, 0, :]
            print('Tracking streamline batch number ', batch_idx + 1, ' out of ', num_batches)

            for t_step in range(self.track_length):

                # Predict streamline direction
                pdf_pred = self.model.predict_on_batch(dwi_inputs)
                if self.tractography_type == 'deterministic':
                    direction_idx_pred = argmax_from_pdf(pdf_pred[:, t_step, :])
                else:
                    direction_idx_pred = sample_from_pdf(pdf_pred[:, t_step, :], 1)[:, 0]

                # Evaluate which streamlines need to be terminated
                if t_step > 0:
                    d_angles = np.array([angles725[direction_idx_pred[p], direction_idx_previous[p]] for p in
                                         range(len(direction_idx_pred))])
                    angle_mask = np.logical_or(angle_mask, d_angles > self.max_angle)
                direction_idx_previous = direction_idx_pred

                odf_entropys = -np.sum(pdf_pred[:, t_step, :] * np.log(pdf_pred[:, t_step, :] + 1e-10), axis=1)
                entropy_mask = np.logical_or(entropy_mask, odf_entropys > self.entropy_th[t_step])
                direction_vec_pred = idx2direction(direction_idx_pred, sphere724)
                EoF_mask = np.logical_or(EoF_mask, direction_idx_pred == sphere724.x.shape[0])
                valids_mask = np.logical_and(np.logical_and(np.logical_and(~EoF_mask, ~entropy_mask), ~angle_mask),
                                             inWM_mask)

                # Calculate the next (x,y,z) location
                # next_positions = next_positions + step_size * direction_vec_pred * np.expand_dims(1 * (~EoF_mask),
                #                                                                                   axis=1) * np.expand_dims(
                #     1 * (~entropy_mask), axis=1) * np.expand_dims(1 * (~angle_mask), axis=1) * np.expand_dims(
                #     1 * (inWM_mask), axis=1)

                next_positions = next_positions + self.step_size * direction_vec_pred * np.expand_dims(1 * (valids_mask), axis=1)
                if sum(1 * valids_mask) == 0:
                    break

                # inWM_mask = np.logical_and(inWM_mask, is_within_mask(2 * next_positions, dilated_wm_mask).astype(bool))
                inWM_mask = np.logical_and(inWM_mask, is_within_mask(next_positions, dilated_wm_mask).astype(bool))

                for k in list(compress(range(len(valids_mask)), valids_mask)):
                    batch_streamlines[k] = np.vstack((batch_streamlines[k], next_positions[k, :]))

                if t_step + 1 < dwi_inputs.shape[1]:
                    dwi_inputs[:, t_step + 1, :] = \
                        eval_volume_at_3d_coordinates(data_handler.dwi, next_positions) - self.dwi_means

            lengths_vec = fiber_lengths(batch_streamlines, [2, 2, 2])
            filtered_out_fibers = [batch_streamlines[k] for k in range(len(batch_streamlines)) if
                                   np.logical_and(lengths_vec[k] > self.min_length, lengths_vec[k] < self.max_length)]
            all_streamlines.append(filtered_out_fibers)

            # if reps + 1 < repetitions:
            #     Loc_seeds = zero_pad_seeds(Loc_seeds_list[reps + 1], len(Loc_seeds_list[reps + 1]), N_time_steps)
            #     out_fibers = list(np.expand_dims(Loc_seeds[i, 0, :], axis=0) for i in range(Loc_seeds.shape[0]))
            #     DW_seeds = np.zeros((Loc_seeds.shape[0], Loc_seeds.shape[1], len(dwi_means)))
            #     DW_seeds[:, 0, :] = eval_volume_at_3d_coordinates(resampled_dwi, Loc_seeds[:, 0, :]) - dwi_means

        tractogram = output_tractogram(all_streamlines)
        return tractogram

    def track(self):
        """
        Organizes the dwi data and runs tractography.
        :return: final out_tractogram - a list containing all tracked streamlines.
        """

        # Set data
        data_handler = self.data_handler
        data_handler.dwi = data_handler.mask_dwi()
        data_handler.dwi = data_handler.resample_dwi()
        data_handler.dwi = data_handler.max_val * data_handler.mask_dwi()
        self.dwi_means = calc_mean_dwi(data_handler.dwi, data_handler.wm_mask)

        # Set random seed points
        seed_mask = self.get_seed_mask(data_handler)
        seed_points = init_seeds(seed_mask, self.num_seeds)

        # partition seeds into batches
        num_batches = int(self.num_seeds / self.track_batch_size)
        if np.mod(self.num_seeds, self.track_batch_size) > 0:
            num_batches += 1

        out_streamlines = self.streamline_tracking(data_handler, seed_points, num_batches)
        out_tractogram = streamlines.tractogram.Tractogram(streamlines=out_streamlines,
                                                           affine_to_rasmm=np.eye(4))
        if self.save_tractogram:
            streamlines.save(tractogram=out_tractogram, filename=join(self.save_dir, 'out_tractogram.tck'))

        return out_tractogram
