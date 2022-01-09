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
from Utils import *
import pandas as pd





def extract_labeled_dcm(input_path):
    data_info={}
    mask_files_path=glob.glob(os.path.join(input_path,'*.nii'))
    mask_file=[sitk.ReadImage(i) for i in mask_files_path]
    series_IDs = sitk.ImageSeriesReader.GetGDCMSeriesIDs(input_path)
    nb_series = len(series_IDs)
    # 通过ID获取该ID对应的序列所有切片的完整路径， series_IDs[0]代表的是第一个序列的ID
    # 如果不添加series_IDs[0]这个参数，则默认获取第一个序列的所有切片路径
    for i in range(len(mask_file)):
        for j in range(nb_series):#
            series_file_names = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(input_path, series_IDs[j])#series_file_names是一个元组，存储了匹配的文件序列
            #print(type(series_file_names))
            series_reader = sitk.ImageSeriesReader()
            series_reader.SetFileNames(series_file_names)
            # 获取该序列对应的3D图像
            image3D = series_reader.Execute()
            if image3D.GetSize()==mask_file[i].GetSize():
                data_info[mask_files_path[i]]=series_file_names
    data = json.dumps(data_info)
    f2 = open('new_json.json', 'w')
    f2.write(data)
    f2.close()       

if __name__=='__main__':
    mask_key_path='/home/wyh21/AI_Lung_node/data_processing/label.csv'
    mask_key=pd.read_csv(mask_key_path)
    