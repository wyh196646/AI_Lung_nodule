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



def detect_dcm_or_nii(patient_path):
    dir=Path(patient_path).parents[0]
    temp=list(map(lambda x:x.suffix,dir.iterdir()))
    if '.dcm' in temp:
        return 'dcm'
    else:
        return 'nii'


if __name__ == '__main__':

    '''
    扫盘程序，这部分代码仅限于我这边这部分数据的处理，其他部分的数据扫盘的时候需要进一步处理
    
    '''
    original_path='/data/wyh_data/影像数据EGFR1第二部分'
    special_path='/data/wyh_data/影像数据EGFR1第二部分/Marked/POSITIVE/3-1marked1'



    res=get_file(original_path)
    temp=filter(lambda x:Path(x).suffix in ['.gz'],res)
    temp=filter(lambda x: 'Mask' in x,res)
    label1path=list(temp)

    special=get_file(special_path)
    temp=filter(lambda x:Path(x).suffix in ['.nii'],special)
    label2path=list(temp)
    label=pd.DataFrame(label1path+label2path,columns=['mask_path'])
    label.drop_duplicates(subset=['mask_path'],inplace=True)
    label.to_csv('/home/wyh21/AI_Lung_node/data_processing/label.csv',index=False)


            
    label['img_type']=label['mask_path'].apply(lambda x:detect_dcm_or_nii(x))
    label['label_type']=label['mask_path'].apply(lambda x:Path(x).suffix[1:])
    label.to_csv('/home/wyh21/AI_Lung_node/data_processing/label.csv',index=False)