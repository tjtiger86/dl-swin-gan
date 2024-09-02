"""
This script performs the first stage of preparation of the Stanford 2D Cine Dataset.

First, why are there two stages? For modularity. The first step dumps out the raw k-space data
from the original ScanArchive files. This step takes quite a while because the binary we use
is the GE product reconstruction. So in fact, it performs a full reconstruction and produces DICOMs in the
Output/ folder. As a byproduct, it also outputs the k-space data in HDF5 format, which is readable by h5py.

The second step reads in the raw k-space data, performs pre-processing necessary for network training
such as coil compression, sensitivity map estimation, and ground truth calculation.
"""

import os
import glob
import argparse
import logging

# define logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main(args):
    logger.info(f'Processing exam data in {args.input_path}')

    # get list of exams
    exam_list = glob.glob(os.path.join(args.input_path, 'Exam*'))

    # go into each exam folder...
    for exam_path in exam_list:
        logger.info(f'  {os.path.split(exam_path)[-1]}...')

        # get list of series
        series_list = glob.glob(os.path.join(exam_path, 'Series*'))

        # go into each series folder...
        for series_path in series_list:
            logger.info(f'    {os.path.split(series_path)[-1]}...')

            logger.info(f"       rm -rf {os.path.join(series_path, 'Output/')}")
            os.system(f"rm -rf {os.path.join(series_path, 'Output/')}")
            logger.info(f"       rm -rf {os.path.join(series_path, 'CineDataGCC/')}")
            os.system(f"rm -rf {os.path.join(series_path, 'CineDataGCC/')}")

    return

def create_arg_parser():
    parser = argparse.ArgumentParser(description="Clear prepped files for Cine Dataset")
    parser.add_argument('--input-path', type=str,
                        default='/mnt/dense/sandino/Studies_StanfordCine/ge', help='Path to input data.')
    # Debug parameters
    parser.add_argument('--dbwrite', action='store_true', help='Write out files for debugging.')
    parser.add_argument('--verbose', action='store_true')
    return parser


if __name__ == '__main__':
    args = create_arg_parser().parse_args()
    main(args)
