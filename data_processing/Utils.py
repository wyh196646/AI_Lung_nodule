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
import json

def detect_dcm_or_nii(patient_path):
    dir=Path(patient_path).parent[0]
    temp=map(lambda x:x.suffix,dir.iterdir())
    if 'dcm' in temp:
        return 'dcm'
    else:
        return 'nii'



def get_file(root_path,all_files=[]):
    '''
    递归函数，遍历该文档目录和子目录下的所有文件，获取其path
    '''
    files = os.listdir(root_path)
    for file in files:
        if not os.path.isdir(root_path + '/' + file):   # not a dir
            all_files.append(root_path + '/' + file)
        else:  # is a dir
            get_file((root_path+'/'+file),all_files)#这里应该是有一个小小的技巧，就是Python传递参数用元组和字典，就可以划分开
            #positional argument 和 keyword argument
    return all_files


# def extract_labeled_from_dcm_or_nii(mask_path,img_type,label_type):
#     '''
#     本函数默认只支持dcm和nii格式的数据
#     输入的是mask的路径，默认mask和影像数据在同一个文件夹下
#     根据文件夹内部的序列进行匹配
#     '''
#     img_path=Path(mask_path).parents[0]

#     if img_type=='dcm':
#         series_IDs = sitk.ImageSeriesReader.GetGDCMSeriesIDs(img_path)
#         #nb_series = len(series_IDs)
#     else:




#     input_path=Path(mask_path).parents[0]

#     if img_type=='dcm':
#         data_info={}
#         mask_files_path=glob.glob(os.path.join(input_path,'*.nii'))
#         mask_file=[sitk.ReadImage(i) for i in mask_files_path]
#         series_IDs = sitk.ImageSeriesReader.GetGDCMSeriesIDs(input_path)
#         nb_series = len(series_IDs)
#         # 通过ID获取该ID对应的序列所有切片的完整路径， series_IDs[0]代表的是第一个序列的ID
#         # 如果不添加series_IDs[0]这个参数，则默认获取第一个序列的所有切片路径
#         for i in range(len(mask_file)):
#             for j in range(nb_series):#
#                 series_file_names = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(input_path, series_IDs[j])#series_file_names是一个元组，存储了匹配的文件序列
#                 #print(type(series_file_names))
#                 series_reader = sitk.ImageSeriesReader()
#                 series_reader.SetFileNames(series_file_names)
#                 # 获取该序列对应的3D图像
#                 image3D = series_reader.Execute()
#                 if image3D.GetSize()==mask_file[i].GetSize():
#                     data_info[mask_files_path[i]]=series_file_names
#     else:
#         img_list=input_path.iterdir()

#         itk_img = sitk.ReadImage('./nifti.nii.gz')
#         img = sitk.GetArrayFromImage(itk_img)

    
    data = json.dumps(data_info)
    f2 = open('new_json.json', 'w')
    f2.write(data)
    f2.close()                


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


def detect_dcm_or_nii(patient_path):
    dir=Path(patient_path).parents[0]
    temp=list(map(lambda x:x.suffix,dir.iterdir()))
    if '.dcm' in temp:
        return 'dcm'
    else:
        return 'nii'