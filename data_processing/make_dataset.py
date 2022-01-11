import os
import shutil
import numpy as np

from scipy.io import loadmat
import numpy as np
import h5py
import pandas
import scipy
from scipy.ndimage.interpolation import zoom
from scipy.ndimage.measurements import label
from skimage import measure
import SimpleITK as sitk
from scipy.ndimage.morphology import binary_dilation,generate_binary_structure
from skimage.morphology import convex_hull_image
import pandas
from multiprocessing import Pool
from functools import partial
import sys
sys.path.append('./preprocessing')
from preprocessing.step1 import step1_python
#from step1 import step1_python
import warnings
import json
from Pathlib import Path


from multiprocessing import Pool




def read_and_convert_nii_to_array(nii_path):
    img = sitk.ReadImage(nii_path)
    img_array = sitk.GetArrayFromImage(img)
    return img_array


def get_context_nodule_coordinate(*params,ratio=1.5):#params一般是6个坐标
    return [calculate_axis_coordinate(x,y) for x,y in zip(params[::2], params[1::2])]

def calculate_axis_coordinate(a,b,ratio):
    return (a+b+ratio*a-ratio*b)//2,(a+b+ratio*b-ratio*a)//2


def classify_nodule_and_relabel(valid_z_pathes,img_array,mask_array):
    '''
    object as the fomal params,the origin num will be changed
    '''
    nodule_start=1
    for i in range(1,len(valid_z_pathes)):
        if valid_z_pathes[i]-valid_z_pathes[i-1]>1:
            nodule_start+=1
            mask_array[valid_z_pathes[i]][ mask_array[valid_z_pathes[i]]==1]=nodule_start
        else:
            if valid_z_pathes[i]-valid_z_pathes[i-1]==1:#这是层数挨着的patch
                if np.isin(1,mask_array[valid_z_pathes[i]]==img_array[valid_z_pathes[i-1]]):#证明两个patch是一个结节的
        
                    mask_array[i][ mask_array[valid_z_pathes[i]]==1]=nodule_start
                else:
                    nodule_start+=1
                    mask_array[valid_z_pathes[i]][mask_array[valid_z_pathes[i]]==1]=nodule_start
    return i


def save_np_array(array,path):#将文件上一级目录下的文件夹创建，然后保存文件夹
    path.parents[0].mkdir(parents=True, exist_ok=True)
    np.save(path,array)
    





def resample(imgs, spacing, new_spacing,order=2):
    if len(imgs.shape)==3:
        new_shape = np.round(imgs.shape * spacing / new_spacing)
        true_spacing = spacing * imgs.shape / new_shape
        resize_factor = new_shape / imgs.shape
        imgs = zoom(imgs, resize_factor, mode = 'nearest',order=order)
        return imgs, true_spacing
    elif len(imgs.shape)==4:
        n = imgs.shape[-1]
        newimg = []
        for i in range(n):
            slice = imgs[:,:,:,i]
            newslice,true_spacing = resample(slice,spacing,new_spacing)
            newimg.append(newslice)
        newimg=np.transpose(np.array(newimg),[1,2,3,0])
        return newimg,true_spacing
    else:
        raise ValueError('wrong shape')


def worldToVoxelCoord(worldCoord, origin, spacing):
     
    stretchedVoxelCoord = np.absolute(worldCoord - origin)
    voxelCoord = stretchedVoxelCoord / spacing
    return voxelCoord

def load_itk_image(filename):
    with open(filename) as f:
        contents = f.readlines()
        line = [k for k in contents if k.startswith('TransformMatrix')][0]
        transformM = np.array(line.split(' = ')[1].split(' ')).astype('float')
        transformM = np.round(transformM)
        if np.any( transformM!=np.array([1,0,0, 0, 1, 0, 0, 0, 1])):
            isflip = True
        else:
            isflip = False

    itkimage = sitk.ReadImage(filename)
    numpyImage = sitk.GetArrayFromImage(itkimage)
     
    numpyOrigin = np.array(list(reversed(itkimage.GetOrigin())))
    numpySpacing = np.array(list(reversed(itkimage.GetSpacing())))
     
    return numpyImage, numpyOrigin, numpySpacing,isflip

def process_mask(mask):
    convex_mask = np.copy(mask)
    for i_layer in range(convex_mask.shape[0]):
        mask1  = np.ascontiguousarray(mask[i_layer])
        if np.sum(mask1)>0:
            mask2 = convex_hull_image(mask1)
            if np.sum(mask2)>1.5*np.sum(mask1):
                mask2 = mask1
        else:
            mask2 = mask1
        convex_mask[i_layer] = mask2
    struct = generate_binary_structure(3,1)  
    dilatedMask = binary_dilation(convex_mask,structure=struct,iterations=10) 
    return dilatedMask


def lumTrans(img):
    lungwin = np.array([-1200.,600.])
    newimg = (img-lungwin[0])/(lungwin[1]-lungwin[0])
    newimg[newimg<0]=0
    newimg[newimg>1]=1
    newimg = (newimg*255).astype('uint8')
    return newimg


def make_dataset(id,label_list=[],image_list=[],prep_folder=''):        
    resolution = np.array([1,1,1])
    image = image_list[id]
    label=label_list[id]
    path=Path(prep_folder)/Path(label[15:]).parents[0]
    path.mkdir(parents=True,exist_ok=True)

    im, m1, m2, spacing = step1_python(image)
    Mask = m1+m2#上面是进行掩膜分割，已经基本调通了，这边的代码尽量不要动，方便在另一边修改接口
    newshape = np.round(np.array(Mask.shape)*spacing/resolution)
    xx,yy,zz= np.where(Mask)
    box = np.array([[np.min(xx),np.max(xx)],[np.min(yy),np.max(yy)],[np.min(zz),np.max(zz)]])
    box = box*np.expand_dims(spacing,1)/np.expand_dims(resolution,1)
    box = np.floor(box).astype('int')
    margin = 5
    extendbox = np.vstack([np.max([[0,0,0],box[:,0]-margin],0),np.min([newshape,box[:,1]+2*margin],axis=0).T]).T
    extendbox = extendbox.astype('int')

    convex_mask = m1
    dm1 = process_mask(m1)
    dm2 = process_mask(m2)
    dilatedMask = dm1+dm2
    Mask = m1+m2
    extramask = dilatedMask - Mask
    bone_thresh = 210
    pad_value = 170
    im[np.isnan(im)]=-2000
    sliceim = lumTrans(im)#分割任务的标签
    sliceim = sliceim*dilatedMask+pad_value*(1-dilatedMask).astype('uint8')
    bones = sliceim*extramask>bone_thresh
    sliceim[bones] = pad_value
    sliceim1,_ = resample(sliceim,spacing,resolution,order=1)
    sliceim2 = sliceim1[extendbox[0,0]:extendbox[0,1],
                extendbox[1,0]:extendbox[1,1],
                extendbox[2,0]:extendbox[2,1]]
    sliceim = sliceim2[np.newaxis,...]
    np.save(path/'image_clean.npy',sliceim)

    label_resample=resample(label,spacing,resolution,order=0)
    np.save(path/'image_segamatation_label.npy',label_resample)#重采样以后的分割标签

    '''
    接下来是切出cube和带边缘信息的cube，EGFR分类任务的标签，我直接做切分以后的

    '''





    #接下来是目标检测的标签




def full_prep(label_list,image_list,output_folder,nproc=150):
    pool = Pool(nproc)
    partial_savenpy = partial(savenpy,label_list=label_list,image_list=image_list,prep_folder=output_folder)
    N = len(label_list)
        #savenpy(1)
    _=pool.map(partial_savenpy,range(N))
    pool.close()
    pool.join()
    print('end preprocessing')

   # f= open(finished_flag,"w+")        




if __name__=='__main__':
    temp=open('/home/wyh21/AI_Lung_node/data_processing/new_json.json')
    bupt=json.load(temp)
    output_path='/data/wyh_data/processed_data'
    label_list=list(bupt.keys())
    image_list=list(bupt.values())
    full_prep(label_list,image_list,output_path)

    print('All dataset maked down')