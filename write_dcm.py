import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import sys
import argparse
import os
from dl_cs.fileio import cfl
import random
import datetime
import numpy as np

import pydicom
from pydicom.dataset import Dataset, FileDataset
from pydicom.uid import ExplicitVRLittleEndian
import pydicom._storage_sopclass_uids
from pydicom.valuerep import PersonName

def write_dicom(pixel_array, filename, seed, study_type, series_num, accel, num_phase, num_slice, im_num, tot_im, slice, phase):

    image = pixel_array.astype(np.uint16)

    print("Setting file meta information...")
    # Populate required values for file meta information

    meta = pydicom.Dataset()
    meta.MediaStorageSOPClassUID = pydicom._storage_sopclass_uids.MRImageStorage
    meta.MediaStorageSOPInstanceUID =  pydicom.uid.generate_uid(entropy_srcs=seed[2])
    meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian  

    ds = Dataset()
    ds.file_meta = meta

    ds.is_little_endian = True
    ds.is_implicit_VR = False

    ds.SOPClassUID = pydicom._storage_sopclass_uids.MRImageStorage
    ds.SOPInstanceUID = pydicom.uid.generate_uid(entropy_srcs=seed[2])
    #print("SOP Instance UID is {}\nMS SOP Instance UID is {}".format(ds.SOPInstanceUID, meta.MediaStorageSOPInstanceUID))
    
    ds.Modality = "MR"
    ds.SeriesInstanceUID = pydicom.uid.generate_uid(entropy_srcs=seed[1])
    ds.StudyInstanceUID = pydicom.uid.generate_uid(entropy_srcs=seed[0])
    ds.FrameOfReferenceUID = pydicom.uid.generate_uid(entropy_srcs=seed[0]+"1")

    ds.StudyID = seed[0][0:10]
    ds.SeriesID = seed[1][0:10]
    
    ds.StudyDescription = "Cardiac CINE " + study_type
    ds.SeriesDescription = "SAX FIESTA Cine BH " + accel
    ds.SeriesNumber = str(series_num)
 
    dt = datetime.datetime.now()

    ds.StudyDate = dt.strftime("%Y%m%d")
    ds.StudyTime = dt.strftime("%H%M%S")
    ds.SeriesDate = dt.strftime("%Y%m%d")
    ds.SeriesTime = dt.strftime("%H%M%S")

    pn = PersonName.from_named_components(
        family_name='lastname',
        given_name='firstname',
        name_suffix='MD'
    )

    ds.PatientBirthDate = dt.strftime("%Y%m%d")
    ds.PatientID = "123456"
    ds.PatientName = pn
    ds.PatientSex="F"
    ds.PatientIdentityRemoved = "YES"

    ds.BitsStored = 16
    ds.BitsAllocated = 16
    ds.SamplesPerPixel = 1
    ds.HighBit = 15

    ds.ImagesInAcquisition = tot_im
    ds.Rows = image.shape[0]
    ds.Columns = image.shape[1]
    ds.InstanceNumber = im_num
    ds.AcquisitionNumber = 1
    ds.CardiacNumberOfImages=num_phase
    ds.HeartRate="60"
   
    ds.RescaleIntercept = "0"
    ds.RescaleSlope = "1"
    ds.PixelSpacing = r"0.8\0.8"
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.PixelRepresentation = 1

    ds.MRAcquisitionType = "2D"
    ds.PulseSequenceName = "SSFP"
    ds.ScanningSequence="GR"
    ds.SequenceVariant = "SS"
    ds.ImageType = "OTHER"

    ds.StackID = "1"
    ds.InStackPositionNumber = slice
    ds.SpacingBetweenSlices="10.0"
    ds.SliceLocation=slice*int(ds.SpacingBetweenSlices)
    ds.SliceThickness="10.0"

    ds.ImagePositionPatient = r"0\0\{}".format(ds.SliceLocation)
    ds.ImageOrientationPatient = r"1\0\0\0\1\0"
    ds.ImageType = r"ORIGINAL\PRIMARY\AXIAL"

    print("Total Images is {}, Instance Number is {}, and InStackPositionNumber is {}".format(ds.ImagesInAcquisition, ds.InstanceNumber, ds.InStackPositionNumber))
    print("Phase is {}, Slices is {}, Slice Location is {}".format(phase, slice, ds.SliceLocation))
    pydicom.dataset.validate_file_meta(ds.file_meta, enforce_standard=True)

    print("Setting pixel data...")
    ds.PixelData = image.tobytes()

    ds.save_as(filename, write_like_original=False)
    return

def normalize(im):
    #return 2**10*np.abs((im-np.mean(im))/(2*np.std(im)))
    im = abs(im)
    minval = np.percentile(im, 40)
    maxval = np.percentile(im, 90)
    im = np.clip(im, minval, maxval)
    return 2**16*(im-np.min(im[:]))/(np.max(im[:]-np.min(im[:])))

def fileParse(file_name, im_num, ext):
    fname = os.path.splitext(file_name)[0]
    fname = fname.split('/')[-1]
    accel = fname.split('_')[-1]
    accel = accel[-8:]
    #return fname+"_slice"+str(slice)+"_phase"+str(phase)+ext, accel
    return str(im_num)+ext, accel

def ZeroPad(im, im_size):

    diff = im_size - np.shape(im)
    pad = []
    for i,d in enumerate(diff): 
        pad.append((int(np.floor(d/2)), int(np.ceil(d/2))))
    return np.pad(im, pad)
    
def Crop_Resize_Im(im, im_size):
    im_ft = np.fft.fftshift(np.fft.fft2(im))
    im_ft = ZeroPad(im_ft, im_size)
    return np.abs(np.fft.ifft2(im_ft))

def calc_patient_position(rel_position):
    #Will do a sample image_orient that came from a SAX acquisitoin
    image_orient = np.array([[0.7, -0.2, 0.7],[ 0.4, 0.8, -0.4]])
    image_orient = np.transpose(image_orient)  
    cross = np.cross(image_orient[:,0], image_orient[:,1])
    R = np.concatenate(image_orient, cross, axis=1)

    return R*rel_position

def main(args):

    file_recon = args.file
    ext = args.ext
    save_dir = args.save_dir
    study_type = args.study_type
    x0,x1 = args.xrange
    y0,y1 = args.yrange
    im_expand_dim = np.array(args.im_size)

    random.seed()
    randstudy = str(int(random.random()*1e10))
    #print("randstudy is {}".format(randstudy))
    
    for i, files in enumerate(file_recon):

        #Need a different rand integer for each series
        random.seed()
        randseries = str(int(random.random()*1e11))
        #print("randseries is {}".format(randseries))
        
        curr_study = cfl.readcfl(files)
        nx, ny, nslice, nmap, nphase, d1, d2, d3 = curr_study.shape
        
        
        series_num = i+1
        im_num = 0
        tot_im = nslice*nphase
        
        """
        curr_image = np.squeeze(curr_study[:,:,:,0,:,:,:,:]) 
        curr_image = normalize(curr_image)
        fname, accel = fileParse(files, 0, 0, ext)
        write_dicom(curr_image, save_dir+fname, [randstudy, randseries], series_num, accel, 1, tot_im)
        """

        #if not os.path.exists(save_dir+study_type+"/"):
        #    os.makedirs(save_dir+study_type+"/")
        
        for k in range(nphase):
            for j in range(nslice):

                random.seed()
                randim = str(int(random.random()*1e10))
  
                #Discard 2nd ESPIRiT map
                curr_image = np.squeeze(curr_study[:,:,j,0,k,:,:,:])
                curr_image = normalize(curr_image)

                #Perform Image Cropping 
                if x0 > 0:
                    curr_image = curr_image[x0:x1, :]
                if y0 > 0:
                    curr_image = curr_image[:, y0:y1]

                #Perform Image Expand
                if im_expand_dim[0] > 0:
                    curr_image = Crop_Resize_Im(curr_image, im_expand_dim)
                
                fname, accel = fileParse(files, im_num, ext)
                
                #write_dicom(curr_image, save_dir+"Series"+str(i+1)+"/"+fname, [randstudy, randseries], series_num, accel, nphase, nslice, im_num, tot_im, j, k)
                write_dicom(curr_image, save_dir+study_type+"_Series"+str(i+1)+"_"+fname, [randstudy, randseries, randim], study_type, series_num, accel, nphase, nslice, im_num, tot_im, j, k)
                im_num = im_num+1
        


def create_arg_parser():
    parser = argparse.ArgumentParser(description="DICOM Conversion")
    parser.add_argument('--file', nargs='+', type=str, required=True, help='file name')
    parser.add_argument('--ext', type=str, default='.dcm', help='extension, default is .dcm')
    parser.add_argument('--save_dir', type=str, default='./', help='default save folder')
    parser.add_argument('--study_type', type=str, default='None', help='study type')
    parser.add_argument('--im_size', nargs=2, type=int, default = (-1, 1), required=False, help='im_size')
    parser.add_argument('--xrange', nargs=2, type=int, default = (-1, -1), required=False, help='x_range')
    parser.add_argument('--yrange', nargs=2, type=int, default = (-1, -1), required=False, help='y_range')

    return parser


if __name__ == '__main__':
    args = create_arg_parser().parse_args(sys.argv[1:])

    main(args)