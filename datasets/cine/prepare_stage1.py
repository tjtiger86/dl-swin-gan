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

# define Orchestra Solo binary
bin_ox = '/home/sandino/bin/OrchestraCineSolo_nPhases_MR27.exe'


def main(args):
    # get list of exams
    exam_list = glob.glob(os.path.join(args.input_path, 'Exam*'))

    # go into each exam folder...
    for exam_name in exam_list:
        logger.info(f'Processing {exam_name}...')

        # get list of series
        exam_path = os.path.join(args.input_path, exam_name)
        series_list = glob.glob(os.path.join(exam_path, 'Series*'))

        # go into each series folder...
        for series_name in series_list:
            logger.info(f'  {series_name}...')

            # get list of all ScanArchive files
            series_path = os.path.join(exam_path, series_name)
            archive_list = glob.glob(os.path.join(series_path, 'ScanArchive*.h5'))

            # if there are multiple ScanArchives, pick the largest one
            if len(archive_list) > 1:
                logger.info('    Found multiple ScanArchives! Choosing largest one...')
                archive_list = sorted(archive_list, key=os.path.getsize)
            archive_name = archive_list[-1]

            # specify output path to put recon files (will be read in stage 2)
            output_path = os.path.join(series_path, 'Output/')
            log_name = os.path.join(series_path, 'OxRecon.log')

            # launch Orchestra Cine Solo
            os.system(f'{bin_ox} --GERecon.InputFile {archive_name} --Cine.ReconPhases {args.nphases} --Cine.InterpMode 0 --GERecon.EnableProcessingArchive --GERecon.TestOutputPath {output_path} --GERecon.SafeParamCollectionSerialization --GERecon.NonInteractive > {log_name}')

    return

def create_arg_parser():
    parser = argparse.ArgumentParser(description="Training script for unrolled MRI recon.")
    parser.add_argument('--input-path', type=str,
                        default='/mnt/dense/sandino/Studies_StanfordCine/combined', help='Path to input data.')
    parser.add_argument('--nphases', type=int, default=20, help='Number of cardiac phases to reconstruct.')
    # Debug parameters
    parser.add_argument('--dbwrite', action='store_true', help='Write out files for debugging.')
    parser.add_argument('--verbose', action='store_true')
    return parser


if __name__ == '__main__':
    args = create_arg_parser().parse_args()
    main(args)
