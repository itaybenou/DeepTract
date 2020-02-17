from config import Parameters
from train import Trainer
from track import Tracker
import argparse
import os
import logging
# import sys
import imp


if __name__ == '__main__':

    # Arguments parsing
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', dest='config', required=False, help='Path to .py configuration file')
    parser.add_argument('--train', action="store_true", required=False, help='Whether to start training phase')
    parser.add_argument('--track', action="store_true", required=False, help='Whether to start inference phase')
    args = parser.parse_args()

    # Logging control
    if args.config is not None:
        log_path = args.config.split('.')[0] + '.log'
    else:
        abspath = os.path.abspath(__file__)
        dname = os.path.dirname(abspath)
        log_path = os.path.join(dname, '.log')

    logging.basicConfig(filename=log_path, filemode='a', format='%(asctime)s %(message)s')
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)
    logger.addHandler(console)

    # Get parameter file from arguments or from default source
    if args.config is not None:
        assert args.config.split('.')[-1] == 'py', 'Config path must be a python script file'
        config_name = os.path.split(args.config)[-1].split('.')[0]
        params = imp.load_source(config_name, args.config).Parameters().params
    else:
        logger.info('No custom config file specified. Using existing config file in {}'
                    .format(dname))
        params = Parameters().params

    # Train model
    if args.train:
        trainer = Trainer(logger=logger, params=params)
        train_performance = trainer.train()

    # Run tractography using a trained model
    if args.track:
        tracker = Tracker(logger=logger, params=params)
        tractogram = tracker.track()
