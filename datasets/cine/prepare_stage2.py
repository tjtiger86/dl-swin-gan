"""
This script performs the second (final) stage of preparation of the Stanford 2D Cine Dataset.

First, why are there two stages? For modularity. The first step dumps out the raw k-space data
from the original ScanArchive files. This step takes quite a while because the binary we use
is the GE product reconstruction. So in fact, it performs a full reconstruction and produces DICOMs in the
Output/ folder. As a byproduct, it also outputs the k-space data in HDF5 format, which is readable by h5py.

The second step reads in the raw k-space data, performs pre-processing necessary for network training
such as coil compression, sensitivity map estimation, and ground truth calculation.
"""

import os
import sys
import glob
import argparse
import logging
import h5py
import numpy as np
from scipy import stats

# custom submodules
sys.path.append(os.path.dirname(__file__))
from utils import coilcomp as cc

# add BART modules
toolbox_path = "/home/sandino/bart/bart-0.6.00/"
os.environ["TOOLBOX_PATH"] = toolbox_path
sys.path.append(os.path.join(toolbox_path, 'python'))
import bart, cfl

# define logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# define global vars
max_slices = 20
max_phases = 30
max_coils = 32
h5_fbase = '/Root/Scan0000/Acquisition%04d/Slice0000/InterpolatedPhase%04d/Echo0000/Channel%04d/RawData'


def load_raw_data(filename):
    """
    Loads raw k-space data from ReconArchive file output by OrchestraCineSolo (see stage 1 for this)
    """

    # Load ReconArchive.h5 file
    h5file = h5py.File(filename, 'r')

    # Get the first view of data so we can know the matrix size
    data = h5file[h5_fbase % (0, 0, 0)]
    nky, nkx = data['real'].shape

    # Initialize k-space data array
    kspace = np.zeros((nkx, nky, max_slices, max_coils, max_phases), dtype=np.complex64)

    # Find these as the data is parsed
    num_slices = num_phases = num_coils = 0

    for sl in range(max_slices):
        for coil in range(max_coils):
            for ph in range(max_phases):
                h5dir = h5_fbase % (sl, ph, coil)
                if h5dir in h5file:
                    data = h5file[h5dir]
                    data = data['real'] + 1j*data['imag']
                    kspace[:, :, sl, coil, ph] = data.T

                    if ph > num_phases-1:
                        num_phases = ph + 1
                    if coil > num_coils-1:
                        num_coils = coil + 1
                    if sl > num_slices-1:
                        num_slices = sl + 1
                else:
                    continue

    return kspace[:, :, :num_slices, :num_coils, :num_phases]


def center(kspace):
    """
    Center k-space data in the case that it was acquired with partial echo
    """

    # Get data dimensions
    nkx, nky, num_slices, num_coils, num_phases = kspace.shape

    # center data (only along kx right now)
    center_line = kspace[:, int(nky / 2) + 1, :, 0, 0]  # (xres, num_slices)
    peak_idxs = np.argmax(np.absolute(center_line), axis=0)
    peak_idx = stats.mode(peak_idxs)[0][0]
    nkx_extra = 2 * (nkx - peak_idx) - nkx
    nkx_extra += nkx_extra % 2  # force to be even

    if nkx_extra > 10:
        logger.info(f'      Detected partial kx acq! Zero-padding ({nkx} -> {nkx + nkx_extra})...')
        pad_dims = (nkx_extra, nky, num_slices, num_coils, num_phases)
        zp = np.zeros(pad_dims, dtype=np.complex64)
        kspace = np.concatenate((zp, kspace), axis=0)

    return kspace


def coil_compress(kspace, num_virtual_coils=6):
    """
    Performs coil compression slice-by-slice on 2D cine data
    """

    # Get data dimensions
    nkx, nky, num_slices, num_coils, num_phases = kspace.shape

    if num_virtual_coils < num_coils:
        kspace_cc = np.zeros((num_slices, num_virtual_coils, num_phases, nky, nkx), dtype=np.complex64)
        # cc code expects (num_channels, num_readout, num_kx)
        d = np.transpose(kspace, [2, 3, 4, 1, 0])
        d = np.reshape(d, [num_slices, num_coils, num_phases*nky, nkx])
        for sl in range(num_slices):
            cc_mat = cc.calc_gcc_weights_c(d[sl], num_virtual_coils)
            dsl = cc.apply_gcc_weights_c(d[sl], cc_mat)
            # above returns (num_vchannels, num_readout, num_kx)
            kspace_cc[sl] = np.reshape(dsl, [num_virtual_coils, num_phases, nky, nkx])
        kspace = kspace_cc
        kspace = np.transpose(kspace, [4, 3, 0, 1, 2])  # transpose back

        return kspace


def ecalib(kspace, num_maps=2, num_calib=20, crop_value=0.1):
    """
    Calls BART's ESPIRiT under the hood to estimate sensitivity maps slice-by-slice.
    """

    # Get data dimensions
    nkx, nky, num_slices, num_coils, num_phases = kspace.shape

    # Compute k-space average
    eps = np.finfo(kspace.dtype).eps
    kspace_avg = np.sum(kspace, axis=-1) / (np.sum(kspace != 0, axis=-1) + eps)

    # Initialize maps array
    maps = np.zeros((nkx, nky, num_slices, num_coils, num_maps), dtype=np.complex64)

    for sl in range(num_slices):
        # Perform ESPIRiT calibration in BART
        maps_sl = bart.bart(1, f'ecalib -S -m {num_maps} -r {num_calib} -c {crop_value}', kspace_avg[:, :, sl, None])
        maps[:, :, sl, :, :] = np.squeeze(maps_sl)

    return maps


def fftmod(array):
    """
    Perform fftmod across spatial dimensions
    """
    array[..., ::2] *= -1
    array[..., ::2, :] *= -1
    array *= -1
    return array


def main(args):
    # create data directories if they don't already exist
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
    if not os.path.exists(os.path.join(args.output_path, 'train')):
        os.makedirs(os.path.join(args.output_path, 'train'))
    if not os.path.exists(os.path.join(args.output_path, 'validate')):
        os.makedirs(os.path.join(args.output_path, 'validate'))
    if not os.path.exists(os.path.join(args.output_path, 'test')):
        os.makedirs(os.path.join(args.output_path, 'test'))

    # determine splits manually for now...
    train_exams = ['Exam2323', 'Exam2406', 'Exam3330',
                   'Exam3331', 'Exam3332', 'Exam3410',
                   'Exam3411', 'Exam3412', 'Exam4873',
                   'Exam4874', 'Exam4905', 'Exam5003',
                   'Exam12026', 'Exam12095', 'Exam12146',
                   'Exam12217', 'Exam12219']
    val_exams = ['Exam2200', 'Exam5050']
    test_exams = ['Exam2216', 'Exam2259', 'Exam2455',
                  'Exam2459', 'Exam7790', 'Exam7791',
                  'Exam7792', 'Exam9965', 'Exam9966']
    all_exams = train_exams + val_exams + test_exams

    logger.info(f'Processing exam data in {args.input_path}')

    # get list of exams
    exam_list = glob.glob(os.path.join(args.input_path, 'Exam*'))

    # go into each exam folder...
    for exam_path in exam_list:
        exam_name = os.path.split(exam_path)[-1]
        logger.info(f'  {exam_name}...')

        # get list of series
        series_list = glob.glob(os.path.join(exam_path, 'Series*'))

        # go into each series folder...
        for series_path in series_list:
            series_name = os.path.split(series_path)[-1]
            logger.info(f'    {series_name}...')

            data_list = glob.glob(os.path.join(series_path, 'Output/Archive/ReconArchive*.h5'))
            if len(data_list) > 0:
                # Shouldn't be more than one of these files...
                file_data = data_list[0]
            else:
                logger.info('      Could not find HDF5 raw data. Did you run prepare_stage1.py?')
                continue

            logger.info('      Loading raw data...')
            kspace = load_raw_data(file_data)

            # Get data dimensions
            nkx, nky, num_slices, num_coils, num_phases = kspace.shape

            if args.verbose:
                logger.info(f'      Detected data dimensions:')
                logger.info(f'        Matrix size: {nkx} x {nky}')
                logger.info(f'        Number of slices: {num_slices}')
                logger.info(f'        Number of coils: {num_coils}')
                logger.info(f'        Number of phases: {num_phases}')

            logger.info('      Pre-processing raw data...')
            kspace = coil_compress(kspace, num_virtual_coils=args.coil_compress)
            kspace = center(kspace)
            maps = ecalib(kspace, num_maps=args.num_maps)

            # Convert to dimension ordering for dl-cs-dynamic
            kspace = np.transpose(kspace, [2, 3, 4, 1, 0])  # [slices, coils, phases, y, x]
            maps = np.transpose(maps, [2, 4, 3, 1, 0])  # [slices, emaps, coils, y, x]
            maps = maps[:, :, :, None, :, :]  # [slices, emaps, coils, 1, y, x]

            # Perform fftmod so we can avoid having to do fftshifts
            kspace = fftmod(kspace)
            maps = fftmod(maps)

            # Generate ground truth
            multicoil_images = np.fft.ifft2(kspace, axes=(-2, -1), norm='ortho')
            target = np.sum(np.expand_dims(multicoil_images, axis=1) * np.conj(maps), axis=2)

            #cfl.writecfl('/home/sandino/kspace', kspace)
            #cfl.writecfl('/home/sandino/maps', maps)
            #cfl.writecfl('/home/sandino/target', target)
            #raise ValueError('stop')

            # Determine path to output hdf5 file
            if exam_name in train_exams:
                folder = 'train'
            elif exam_name in val_exams:
                folder = 'validate'
            else:
                folder = 'test'

            # write out HDF5 file for entire volume
            h5_name = f'{exam_name}_{series_name}.h5'
            filename = os.path.join(args.output_path, folder, h5_name)
            with h5py.File(filename, 'w') as hf:
                hf.create_dataset('kspace', data=kspace)
                hf.create_dataset('maps', data=maps)
                hf.create_dataset('target', data=target)

        logger.info(' ')

    return


def create_arg_parser():
    parser = argparse.ArgumentParser(description="Training script for unrolled MRI recon.")
    parser.add_argument('--input-path', type=str,
                        default='/mnt/dense/sandino/Studies_StanfordCine/combined', help='Path to input data.')
    parser.add_argument('--output-path', type=str,
                        default='/mnt/dense/sandino/TorchData/stanfordCine', help='Path to output data.')
    # Data parameters
    parser.add_argument('--coil_compress', type=int, default=8, help='Number of virtual channels.')
    parser.add_argument('--num_maps', type=int, default=2, help='Number of sets of ESPIRiT maps.')
    # Debug parameters
    parser.add_argument('--dbwrite', action='store_true', help='Write out files for debugging.')
    parser.add_argument('--verbose', action='store_true')
    return parser


if __name__ == '__main__':
    args = create_arg_parser().parse_args()
    main(args)
