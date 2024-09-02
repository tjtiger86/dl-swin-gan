import numpy as np
import os
from pathlib import Path
import sys
import glob
import argparse
import scripts.reconstruct_h5 as recon
import folder_param as fp

def main(args):

    data_directory = args.data_dir
    write_directory = args.out_directory
    files = glob.glob(os.path.join(data_directory, '*.h5'))
    config = args.config_file

    #Generating subfolder based on checkpoint file
    ckpt_file = args.ckpt
    path_ckpt = Path(ckpt_file).parts

    if args.sub_directory == 1:

        sub_folder = ""
        for path in path_ckpt:

            param = fp.folder_to_parameter(path, 0, config)

            if param:
                sub_folder = path

        out_dir = os.path.join(write_directory, sub_folder)
        print(out_dir)
        args.out_directory = out_dir

        if not os.path.exists(out_dir):
            print("Create Sub-directory")
            os.makedirs(out_dir)

    for file in files:

        args.file = file
        recon.main(args)

def create_arg_parser():

    parser = argparse.ArgumentParser(description="Batch Recon")
    parser.add_argument('--data-dir', type=str, required=True, help='Directory of data files')
    parser.add_argument('--out-directory', type=str, required=True, help='Output Directory')
    parser.add_argument('--sub-directory', type=int, default=1, help='create subfolder for ouptut directory, 0 no, 1 yes')
    parser.add_argument('--model', type=str, required=True, help='RES (resnet) or SE (squeeze excitation) ')
    parser.add_argument('--acceleration', type=int, default=1, help='Undersampling Accelration Factor')
    parser.add_argument('--ckpt', type=str, required=True, help='Model checkpoint file')
    parser.add_argument('--batch-size', type=int, default=1, help='Model checkpoint file')
    parser.add_argument('--config-file', type=str, required=True, help='Training config file (yaml)')
    parser.add_argument('--device', type=int, default=-1, help='GPU device')
    parser.add_argument('--multi-gpu', action='store_true', help='Uses multiple GPUs for inference (overrides device flag)')
    parser.add_argument('--verbose', action='store_true', help='Turn on debug statements')

    return parser


if __name__ == '__main__':

    args = create_arg_parser().parse_args(sys.argv[1:])
    main(args)
