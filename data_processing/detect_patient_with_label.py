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

def extract_labeled_from_dcm_or_nii(compose_info:tuple):
    mask_path,img_type,label_type=compose_info
    '''
    本函数默认只支持dcm和nii格式的数据
    输入的是mask的路径，默认mask和影像数据在同一个文件夹下
    根据文件夹内部的序列进行匹配
    '''
    #label不管是啥样的，都是一样的读法，不用区分
    input_path=Path(mask_path).parents[0]
    try:
        mask_file=sitk.ReadImage(mask_path)
    except:
        return {'unvalid':mask_path}#为了后面过滤掉无效的数据的一种取巧操作，不得已而为之
        #return 
    else:

        data_info={}
        #根据单元测试，必须对label最后一个维度消融掉，之前那个怎么匹配得上的，是有问题的


        if img_type=='dcm':
            series_IDs = sitk.ImageSeriesReader.GetGDCMSeriesIDs(str(input_path))
            for i in range(len(series_IDs)):
                    series_file_names = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(str(input_path), series_IDs[i])#series_file_names是一个元组，存储了匹配的文件序列
            #print(type(series_file_names))
                    if series_file_names:
                        series_reader = sitk.ImageSeriesReader()
                        series_reader.SetFileNames(series_file_names)
                        # 获取该序列对应的3D图像
                        try:
                            image3D = series_reader.Execute()#有一些空头文件必须处理掉
                        except:
                            return  {'unvalid':mask_path}
                            #return 
                        else:
                            if image3D.GetSize()==mask_file.GetSize()[0:3]:
                                data_info[mask_path]=series_file_names
                                break
            #nb_series = len(series_IDs)
        else:
            files_path=glob.glob(os.path.join(input_path,'*.nii.gz'))
            image_path=list(filter(lambda x:'Mask' not in x,files_path))#过滤掉标签信息
            for i in image_path:
                image_3D=sitk.ReadImage(str(i))
                if image_3D.GetSize()==mask_file.GetSize()[0:3]:
                    data_info[mask_path]=[i]
                    break
        return data_info



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
    mask_key=label
    mask_path,img_type,label_type=convert_dataframe_to_serveral_list(mask_key)
    compose_info=list(zip(mask_path,img_type,label_type))
    res=[]
    with Pool(150) as p:
        res=p.map(extract_labeled_from_dcm_or_nii,compose_info)

    save={}
    for i in res:
        save.update(i)
        
    save.pop('unvalid')
    data = json.dumps(save)
    f2 = open('new_json.json', 'w')
    f2.write(data)
    f2.close()  
    