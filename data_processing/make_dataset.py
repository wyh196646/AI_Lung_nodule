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
from multiprocessing import Pool

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


def savenpy(id,label_list,image_list,prep_folder):        
    resolution = np.array([1,1,1])
    image = image_list[id]
    label=label_list[id]
    #label = annos[annos[:,0]==name]
    #label = label[:,[3,1,2,4]].astype('float')
    #应该是用相对路径更好一些,但是也没有办法了，这里已经搞成绝对路径了，后面再说吧，路径不太好弄
    #这里
    im, m1, m2, spacing = step1_python(img)



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
    sliceim = lumTrans(im)
    sliceim = sliceim*dilatedMask+pad_value*(1-dilatedMask).astype('uint8')
    bones = sliceim*extramask>bone_thresh
    sliceim[bones] = pad_value
    sliceim1,_ = resample(sliceim,spacing,resolution,order=1)
    sliceim2 = sliceim1[extendbox[0,0]:extendbox[0,1],
                extendbox[1,0]:extendbox[1,1],
                extendbox[2,0]:extendbox[2,1]]
    sliceim = sliceim2[np.newaxis,...]
    np.save(os.path.join(prep_folder,name+'_clean.npy'),sliceim)

    
    # if len(label)==0:
    #     label2 = np.array([[0,0,0,0]])
    # elif len(label[0])==0:
    #     label2 = np.array([[0,0,0,0]])
    # elif label[0][0]==0:
    #     label2 = np.array([[0,0,0,0]])
    # else:
    #     haslabel = 1
    #     label2 = np.copy(label).T
    #     label2[:3] = label2[:3][[0,2,1]]#这里应该是交换 x y z坐标
    #     label2[:3] = label2[:3]*np.expand_dims(spacing,1)/np.expand_dims(resolution,1)
    #     label2[3] = label2[3]*spacing[1]/resolution[1]
    #     label2[:3] = label2[:3]-np.expand_dims(extendbox[:,0],1)
    #     label2 = label2[:4].T
    # np.save(os.path.join(prep_folder,name+'_label.npy'),label2)




def full_prep(label_list,image_list,output_folder,nproc=150):
    pool = Pool(nproc)
    partial_savenpy = partial(savenpy,label_list= label_list,file_list=image_list,prep_folder=output_folder)
    N = len(label_list)
        #savenpy(1)
    _=pool.map(partial_savenpy,range(N))
    pool.close()
    pool.join()
    print('end preprocessing')

   # f= open(finished_flag,"w+")        




if __name__=='__main__':
    temp=open('new_json.json')
    bupt=json.load(temp)
    output_path='/data/wyh_data/processed_data'
    label_list=bupt.keys()
    image_list=bupt.values()
    full_prep(label_list,image_list,output_path)

    print('All dataset maked down')