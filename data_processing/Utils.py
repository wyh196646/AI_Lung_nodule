import glob
import shutil
import SimpleITK as sitk
import numpy as np
import os
from collections import Counter
from multiprocessing import Pool
from numpy.lib.npyio import save
from numpy.lib.shape_base import _dstack_dispatcher
from numpy.lib.type_check import imag
from pathlib import Path
import pathlib
import multiprocessing
import math
import functools
from functools import partial
import pydicom


def get_file(root_path,all_files=[]):
    '''
    递归函数，遍历该文档目录和子目录下的所有文件，获取其path
    '''
    files = os.listdir(root_path)
    for file in files:
        if not os.path.isdir(root_path + '/' + file):   # not a dir
            all_files.append(root_path + '/' + file)
        else:  # is a dir
            get_file((root_path+'/'+file),all_files)
    return all_files

def scan_base_folder(root_path,all_files=[]):

    files = os.listdir(root_path)
    for file in files:
        temp_path = root_path + '/' + file
        # if not have_subfolder(temp_path):
        if not have_subfolder(temp_path) and Path:
            all_files.append(temp_path)
        else:
            get_file((temp_path),all_files)  


    return all_files

def have_subfolder(path):#判断是否有子文件夹
    '''
    这里有一定的假设前提，就是说，这个文件夹下完全没有子目录
    才返回True
    '''
    for x in os.listdir(path):
        if  os.path.isdir(path+'/'+x):
            return True
    return False


def load_scan(filelist):
    #slices = [dicom.read_file(path + '/' + s) for s in os.listdir(path)]
    slices = [pydicom.dcmread(s) for s in filelist]
    slices.sort(key = lambda x: float(x.ImagePositionPatient[2]))
    try:
        slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])
    except:
        slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)
        
    for s in slices:
        s.SliceThickness = slice_thickness
        
    return slices


def search_label_(path):
    valid_label=[]
    for i in get_file(path):#用来搜索数据,主要是用来搜索标注信息,
        if 'dcm' not in i and 'txt' not in i and 'LOG' not in i and '.DS_Store' not in i:
            print(i)

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


def read_and_convert_nii_to_array(nii_path):
    img = sitk.ReadImage(nii_path)
    img_array = sitk.GetArrayFromImage(img)
    return img_array


def static_suffix(res=[]):
    suffix_list=[]#用来统计suffix
    for i in res:
        if Path(i).suffix not in suffix_list:
            suffix_list.append(Path(i).suffix)
    return suffix_list