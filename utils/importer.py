# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

#Based on: https://www.kaggle.com/gzuidhof/full-preprocessing-tutorial

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import pydicom
import os
import scipy.ndimage
import matplotlib.pyplot as plt

from skimage import measure, morphology
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# Some constants 
INPUT_FOLDER = os.environ['BOWL2017_SAMPLE_FOLDER']
print("sample images folder: {}".format(INPUT_FOLDER))

MIN_BOUND = -1000.0
MAX_BOUND = 400.0
PIXEL_MEAN = 0.25
    
def normalize(image):
    image = (image - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
    image[image>1] = 1.
    image[image<0] = 0.
    return image

def zero_center(image):
    image = image - PIXEL_MEAN
    return image

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

def load_patient_hu_image(patient_index=0):
    p = get_patients()
    slices = load_scan(p[patient_index])
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
    
def plot_image_slice(image,slice_index=80):
    #plt.hist(hu.flatten(), bins=80, color='c')
    plt.imshow(image[slice_index], cmap=plt.cm.gray)
    plt.show()
    
    
def largest_label_volume(im, bg=-1):
    vals, counts = np.unique(im, return_counts=True)

    counts = counts[vals != bg]
    vals = vals[vals != bg]

    if len(counts) > 0:
        return vals[np.argmax(counts)]
    else:
        return None
    
def segment_lung_mask(image, fill_lung_structures=True):
    
    # not actually binary, but 1 and 2. 
    # 0 is treated as background, which we do not want
    binary_image = np.array(image > -320, dtype=np.int8)+1
    labels = measure.label(binary_image)
    
    # Pick the pixel in the very corner to determine which label is air.
    #   Improvement: Pick multiple background labels from around the patient
    #   More resistant to "trays" on which the patient lays cutting the air 
    #   around the person in half
    background_label = labels[0,0,0]
    
    #Fill the air around the person
    binary_image[background_label == labels] = 2
    
    
    # Method of filling the lung structures (that is superior to something like 
    # morphological closing)
    if fill_lung_structures:
        # For every slice we determine the largest solid structure
        for i, axial_slice in enumerate(binary_image):
            axial_slice = axial_slice - 1
            labeling = measure.label(axial_slice)
            l_max = largest_label_volume(labeling, bg=0)
            
            if l_max is not None: #This slice contains some lung
                binary_image[i][labeling != l_max] = 1

    
    binary_image -= 1 #Make the image actual binary
    binary_image = 1-binary_image # Invert it, lungs are now 1
    
    # Remove other air pockets insided body
    labels = measure.label(binary_image, background=0)
    l_max = largest_label_volume(labels, bg=0)
    if l_max is not None: # There are air pockets
        binary_image[labels != l_max] = 0
 
    return binary_image    
    
def prepare_image_for_processing(patient_index):
    hu_image,slices = load_patient_hu_image(patient_index=patient_index)
    hu_image_resampled,new_spacing,old_spacing = resample(hu_image,slices)
    segmented_lungs_image = segment_lung_mask(hu_image_resampled, fill_lung_structures=True)
    normalized = normalize(segmented_lungs_image)
    image_ready_for_process = zero_center(normalized)    
    return image_ready_for_process
    
def test_resamle():
    hu_image,slices = load_patient_hu_image()
    hu_image_resampled,new_spacing,old_spacing = resample(hu_image,slices)
    return hu_image_resampled,new_spacing,old_spacing

def test_2d_plot(slice_index=80):
    hu_image,slices = load_patient_hu_image()
    hu_image_resampled,new_spacing,old_spacing = resample(hu_image,slices)
    plot_image_slice(image=hu_image_resampled,slice_index=slice_index)    

def test_3d_plot(threshold=400):
    hu_image,slices = load_patient_hu_image()
    hu_image_resampled,new_spacing,old_spacing = resample(hu_image,slices)
    plot_3d_image(image=hu_image_resampled,threshold=threshold)
    
def test_3d_segmen_plot(fill_lung_structures=True):
    hu_image,slices = load_patient_hu_image()
    hu_image_resampled,new_spacing,old_spacing = resample(hu_image,slices)
    segmented_lungs_image = segment_lung_mask(hu_image_resampled, fill_lung_structures)
    plot_3d_image(image=segmented_lungs_image,threshold=0)    
    
def test_3d_segmen_final_plot(fill_lung_structures=True):
    image = prepare_image_for_processing(patient_index=0)
    plot_3d_image(image=image,threshold=0)   


    
    
    
    
    
    

    

    
    


    


    
