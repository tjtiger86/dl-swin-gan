"""
Script for extracting sensitivty maps from GE's ASSET files
"""

import os, sys
import glob
import numpy as np
import argparse

# import custom libraries
from utils import cfl
from utils import fftc

# import orchestra sdk libraries
sys.path.append('/home/sdkuser/orchestra')
from GERecon import AssetFile, RawFile, Archive, Transformer, Asset


def get_scanarchive_filename():
    """returns the name of ScanArchive if available"""
    filename_list = glob.glob(os.path.join(os.getcwd(), 'ScanArchive*.h5'))
    filename_list = sorted(filename_list, key=os.path.getsize)
    if len(filename_list) > 0:
        return filename_list[-1]
    else:
        return None


def get_asset_calibration_filename(archive):
    """returns the Asset calibration file, if available"""

    # Read ScanArchive header
    hdr = archive.Header()

    # Get calibration directory (actually, this is just the exam number)
    exam_number = str(np.int(hdr["rdb_hdr_exam"]["ex_no"]))
    calibration_dir = os.path.join(os.getcwd(), exam_number)

    # Get the calibration UID
    cal_uid = np.int(hdr["rdb_hdr_ps"]["calUniqueNo"]) // 100
    if cal_uid == 0:  # then we probably have a DV26 scan archive
        cal_uid = np.int(hdr["rdb_hdr_rec"]["rdb_hdr_coilConfigUID"]) // 100
    cal_uid = str(cal_uid)

    # Get the name of the calibration file
    if os.path.isdir(calibration_dir):
        file_list = sorted(os.listdir(calibration_dir), reverse=True)  # take one with highest index
        for file_item in file_list:
            base, ext = os.path.splitext(file_item)
            if (base.startswith('Asset-ID' + cal_uid) or  # maybe we have DV26
                    base.startswith('Asset-Coil' + cal_uid)) and ext == '.h5':
                return os.path.join(calibration_dir, file_item)

    return None


def data_build(archive):
    # Extract metadata from ScanArchive
    metadata = archive.Metadata()
    nkx = metadata["acquiredXRes"]
    nky = metadata["acquiredYRes"]
    num_coils = metadata["numChannels"]
    num_ctrl = metadata["controlCount"]
    num_phases = metadata["passes"]  # number of temporal phases

    # Extract number of slices from MRImg header
    hdr = archive.Header()
    num_slices = int(hdr['rdb_hdr_image']['slquant'])

    # For keeping track of the phase number
    ph = 0

    # Declare raw data array
    dims = [num_phases, num_coils, num_slices, nky, nkx]
    kspace_data = np.zeros(dims, dtype=np.complex64)

    # Loop over all of the control packets in ScanArchive
    for _ in range(num_ctrl):
        # Retrieve next control packet
        control = archive.NextControl()

        # This indicates a new k-space line
        if control['packetType'] == 'ProgrammableControlPacket':
            # Get view and slice number
            ky = control['viewNum'] - 1
            sl = control['sliceNum']

            # Retrieve k-space data from current packet
            next_frame = np.squeeze(archive.NextFrame()).T

            # Save into k-space data array
            kspace_data[ph, :, sl, ky, :] = next_frame

        # This indicates the end of a phase
        elif control['packetType'] == 'ScanControlPacket':
            # Increment the phase number
            ph += 1

        else:
            raise ValueError('Type of control packet not recognized!')

    return kspace_data


def maps_build(archive):
    # Extract metadata from ScanArchive
    metadata = archive.Metadata()
    image_x_res = metadata["imageXRes"]
    image_y_res = metadata["imageYRes"]
    num_coils = metadata["numChannels"]

    # Extract number of slices from MRImg header
    hdr = archive.Header()
    num_slices = int(hdr['rdb_hdr_image']['slquant'])

    # Extract ASSET parameters
    asset_params = archive.AssetFileParams()

    # Enforce size of sensitivity maps to be size of output of asset.Unalias()
    asset_params['AcquiredXRes'] = image_x_res
    asset_params['AcquiredYRes'] = image_y_res

    # Initialize AssetFile object to be able to extract sensitivity maps
    cal_filename = get_asset_calibration_filename(archive)
    if cal_filename is not None:
        cal_file = AssetFile(asset_params, cal_filename)
    else:
        raise ValueError('No sensitivity map file available for this ScanArchive.')

    # Initialize numpy array to hold maps
    maps = np.zeros((num_coils, num_slices, image_y_res, image_x_res), dtype=np.complex64)

    # Read maps slice-by-slice
    for slice_num in range(num_slices):
        phase_num = 0  # sensitivity maps shouldnt change across phase number
        corners = archive.AcquiredCorners(slice_num, phase_num)
        maps[:, slice_num, :, :] = cal_file.ExtractSlice(corners).swapaxes(0, 2)

    return maps


def asset_recon(kspace, archive, skip_fermi_filter=True):
    # Extract metadata from ScanArchive
    metadata = archive.Metadata()
    image_x_res = metadata["imageXRes"]
    image_y_res = metadata["imageYRes"]

    # Get calibration filename
    cal_filename = get_asset_calibration_filename(archive)

    # Modify transformer parameters in order to skip k-space filtering
    transformer_params = archive.TransformerParams()
    if skip_fermi_filter:
        print('Skipping fermi filter...')
        transformer_params['SkipFermiFilter'] = skip_fermi_filter

    # Prepare ScanArchive operators
    transform = Transformer(transformer_params)
    asset = Asset(archive.AssetParams(), cal_filename)

    # Get data dimensions
    num_phases, num_coils, num_slices, nky, nkx = kspace.shape

    # Initialize image array
    combined_images = np.zeros((num_phases, num_slices, image_y_res, image_x_res), dtype=np.complex64)

    for slice_num in range(num_slices):

        for phase_num in range(num_phases):
            # Get image corners
            corners = archive.AcquiredCorners(slice_num, phase_num)

            # Need to track image for every channel per slice to combine together into one image later
            channel_images = np.zeros([image_x_res, image_y_res, num_coils], dtype=np.complex64)

            for coil in range(num_coils):
                # Transform a slice of kSpace for a single channel
                transformed_image = transform.Execute(kspace[phase_num, coil, slice_num].T)

                # This is Homodyne data, so resize channel images array to handle the high pass and low pass data
                # Note this only happens once, this first time through the loop
                if transformed_image.ndim == 3 and channel_images.ndim == 3:
                    channel_images = np.zeros([image_x_res, image_y_res, num_coils, 2], dtype=np.complex64)

                # Save the transformed image
                channel_images[:, :, coil] = transformed_image

            # Combine and unalias all the images of the same slice but different channels together into one image,
            # so this outputs an array orgnaizd as ImageXRes x ImageYRes
            combined_image = asset.Unalias(channel_images, corners)
            combined_images[phase_num, slice_num, :, :] = combined_image.T

    return combined_images


def center_crop(data, dims, shapes):
    for i, dim in enumerate(dims):
        assert 0 < shapes[i] <= data.shape[dim]
        idx_start = (data.shape[dim] - shapes[i]) // 2
        data = data.swapaxes(0, dim)
        data = data[idx_start:(idx_start+shapes[i])]
        data = data.swapaxes(0, dim)

    return data


def kspace_build(images, maps, archive):
    # images: num_phases, num_slices, image_y_res, image_x_res
    # maps  : num_coils, num_slices, image_y_res, image_x_res

    # Extract metadata from ScanArchive
    metadata = archive.Metadata()
    nkx = metadata["acquiredXRes"]
    nky = 2 * metadata["acquiredYRes"]

    # Undo modulation in the image domain
    images[..., ::2] *= -1.0
    images[..., ::2, :] *= -1.0

    # Multiply images with maps to get multi-channel images
    multichannel_images = images[:, None, :, :, :] * maps[None, :, :, :, :]

    # Convert images to k-space
    kspace = fftc.fft2c(multichannel_images).astype(np.complex64)

    # Crop to the original matrix size
    kspace = center_crop(kspace, [3, 4], [nky, nkx])

    return kspace


def main(args):
    # Go to directory containing ScanArchive
    os.chdir(args.directory)

    # Get name of ScanArchive
    file_archive = get_scanarchive_filename()
    print('Reading %s...' % file_archive)

    # Load ScanArchive object
    # note: This also creates a folder in the current working directory
    # named as the exam number. This folder contains auxiliary files
    # which contain ASSET, noise statistics, Pure info, etc...
    archive = Archive(file_archive)

    print('Loading k-space data from ScanArchive...')
    kspace = data_build(archive)

    print('Performing ASSET recon...')
    images = asset_recon(kspace, archive)

    print('Loading sensitivity maps from ScanArchive...')
    maps = maps_build(archive)

    print('Simulating fully-sampled k-space data...')
    fully_sampled_kspace = kspace_build(images, maps, archive)

    print('Writing out data in CFL format...')
    cfl.write(os.path.join(args.directory, 'kspace_us'), kspace)
    cfl.write(os.path.join(args.directory, 'kspace_fs'), fully_sampled_kspace)
    cfl.write(os.path.join(args.directory, 'mps'), maps)
    cfl.write(os.path.join(args.directory, 'im.asset'), images)

    return


def create_arg_parser():
    parser = argparse.ArgumentParser(description="Script for preparing perfusion ScanArchive data.")
    parser.add_argument('--directory', type=str, required=True, help='Path to folder containing ScanAchive.')
    # Debug parameters
    parser.add_argument('--dbwrite', action='store_true', help='Write out files for debugging.')
    parser.add_argument('--verbose', action='store_true')
    return parser


if __name__ == '__main__':
    args = create_arg_parser().parse_args()
    main(args)
