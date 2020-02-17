from utils.test_utils import *
from utils.Network import *
from utils.data_handling import *
from os.path import join
from keras.models import model_from_json
import logging


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

        self.data_handler = DataHandler(self.params)
        self.trained_model_dir = self.params['trained_model_dir']
        self.model = None
        self.get_model()
        self.data_handler = DataHandler(self.params, mode='track')

        self.tractography_type = self.params['tractography_type']
        self.num_seeds = self.params['num_seeds']
        self.step_size = self.params['step_size']
        self.max_angle = self.params['max_angle']
        self.max_length = self.params['max_length']
        self.track_length = self.model.input_shape[1]
        self.entropy_params = self.params['entropy_params']
        self.entropy_th = calc_entropy_threshold(self.entropy_params, self.track_length)

    def get_model(self):
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

    def test_scheme(self, data_loader):
        """
        Performs test iterations on the data loader images. If display mode is enabled - shows prediction images.

        :param data_loader: Test data loader
        :return: final scores - average scores on test data set. If labels doesn't exist - return empty dict.
        """
        scores = []
        steps = len(data_loader)
        with torch.no_grad():

            with tqdm(total=steps) as progress:
                for data in data_loader:
                    if self.labels_exist:
                        images, labels, ids = data
                    else:
                        images, ids = data

                    preds = self.model(images.to(self.device))
                    preds = preds.data.cpu().numpy()
                    for i, pred in enumerate(preds):
                        if self.labels_exist:
                            label = tensor_to_image(labels[i])
                            score = get_prediction_scores(pred, label, tuple(np.array(ids)[:, i]),
                                                          self.params['num_classes'])
                            scores.append(list(score.values())[1:])  # first indices is image id
                            if self.display_on:
                                plot_seg_result(tensor_to_image(images[i]), ids[0][i], tensor_to_image(pred),
                                                label, '%0.3f' % score['dice_score'])

                        elif self.display_on:
                            plot_seg_result(tensor_to_image(images[i]), ids[0][i], tensor_to_image(pred), None)

                    progress.update()
            # Calculate final average scores
            final_scores = dict()
            if self.labels_exist:
                avg_scores = np.mean(np.array(scores), axis=0)
                self.logger.info('Total number of test samples: {}'.format(scores.__len__()))
                for metric, score_value in zip(list(score.keys())[1:], avg_scores):
                    self.logger.info('Average {}: {}'.format(metric, '%0.3f' % np.mean(score_value)))
                    final_scores[metric] = score_value

        return final_scores

    def test(self):

        # Initialize data loader
        data_loader = SegmentationDataLoader(self.params, logger=self.logger, mode='test')
        # Evaluate predictions and results
        scores = self.test_scheme(data_loader())

        return scores
