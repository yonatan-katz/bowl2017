# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import pydicom
import os
import scipy.ndimage
import matplotlib.pyplot as plt

from skimage import measure, morphology
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# Some constants 
#INPUT_FOLDER = '/home/yonic/repos/bowl2017/data/sample_images/'
INPUT_FOLDER = os.environ['BOWL2017_SAMPLE_FOLDER']
print("sample images folder: {}".format(INPUT_FOLDER))



def get_patients():
    patients = os.listdir(INPUT_FOLDER)
    return patients

def make_patient_scan_path(patient):
    return os.path.join(INPUT_FOLDER,patient)

def load_scan(patient):
    path = make_patient_scan_path(patient)
    slices = [pydicom.read_file(path + '/' + s) for s in os.listdir(path)]
    slices.sort(key = lambda x: float(x.ImagePositionPatient[2]))
    try:
        slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])        
    except:
        slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)
        
    print ("slice_thickness: {}".format(slice_thickness))
    for s in slices:        
        s.SliceThickness = slice_thickness
    
    return slices

def get_pixels_hu(slices):
    image = np.stack([s.pixel_array for s in slices])
    # Convert to int16 (from sometimes int16), 
    # should be possible as values should always be low enough (<32k)
    image = image.astype(np.int16)

    # Set outside-of-scan pixels to 0
    # The intercept is usually -1024, so air is approximately 0
    image[image == -2000] = 0
    
    # Convert to Hounsfield units (HU)
    for slice_number in range(len(slices)):
        
        intercept = slices[slice_number].RescaleIntercept
        slope = slices[slice_number].RescaleSlope
        
        if slope != 1:
            image[slice_number] = slope * image[slice_number].astype(np.float64)
            image[slice_number] = image[slice_number].astype(np.int16)
            
        image[slice_number] += np.int16(intercept)
    
    return np.array(image, dtype=np.int16)


def resample(image, scan, new_spacing=[1,1,1]):
    # Determine current pixel spacing
    spacing_list = [scan[0].SliceThickness]
    spacing_list.extend(scan[0].PixelSpacing)
    old_spacing = np.array(spacing_list, dtype=np.float32)

    #spacing = np.array([scan[0].SliceThickness] + scan[0].PixelSpacing, dtype=np.float32)

    resize_factor = old_spacing / new_spacing
    new_real_shape = image.shape * resize_factor
    new_shape = np.round(new_real_shape)
    real_resize_factor = new_shape / image.shape
    new_spacing = old_spacing / real_resize_factor
    
    image = scipy.ndimage.interpolation.zoom(image, real_resize_factor, mode='nearest')
    
    return image, new_spacing, old_spacing

def load_patient_hu_image():
    p = get_patients()
    slices = load_scan(p[0])
    hu_image = get_pixels_hu(slices)
    return hu_image,slices

def plot_3d_image(image, threshold=-300):    
    # Position the scan upright, 
    # so the head of the patient would be at the top facing the camera
    p = image.transpose(2,1,0)   
    
    verts, faces,normals,values = measure.marching_cubes(p, threshold)

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Fancy indexing: `verts[faces]` to generate a collection of triangles
    mesh = Poly3DCollection(verts[faces], alpha=0.70)
    face_color = [0.45, 0.45, 0.75]
    mesh.set_facecolor(face_color)
    ax.add_collection3d(mesh)

    ax.set_xlim(0, p.shape[0])
    ax.set_ylim(0, p.shape[1])
    ax.set_zlim(0, p.shape[2])

    plt.show()
    
def plot_image_slice(hu,slice_index=80):
    #plt.hist(hu.flatten(), bins=80, color='c')
    plt.imshow(hu[slice_index], cmap=plt.cm.gray)
    plt.show()
    
def test_resamle():
    hu_image,slices = load_patient_hu_image()
    hu_image_resampled,new_spacing,old_spacing = resample(hu_image,slices)
    return hu_image_resampled,new_spacing,old_spacing

def test_2d_plot():
    hu_image,slices = load_patient_hu_image()
    hu_image_resampled,new_spacing,old_spacing = resample(hu_image,slices)
    plot_image_slice(hu_image_resampled)    

def test_3d_plot():
    hu_image,slices = load_patient_hu_image()
    hu_image_resampled,new_spacing,old_spacing = resample(hu_image,slices)
    plot_3d_image(hu_image_resampled)
    
    


    


    
