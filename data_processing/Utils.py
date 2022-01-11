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
import scipy
from skimage import measure, morphology
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.pyplot as plt
import collections

from ipywidgets import interact, interactive
from ipywidgets import widgets

def count_values(data:np.array):
    unique, counts =np.unique(data, return_counts=True)
    return dict(zip(unique, counts))




def convert_series_to_list(series):
    '''
    将一个Series对象转换成list
    '''
    return series.values.tolist()


def convert_dataframe_to_serveral_list(data):
    '''
    外面的变量数必须对得上列的数量才行
    '''
    temp=[]
    for x in data.columns:
        temp.append(data[x].values.tolist())
    return temp

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
          
          

#mask匹配结节部分差不多了，接下来就是看cube
# mask_path,img_type,label_type=convert_dataframe_to_serveral_list(mask_key)
# compose_info=list(zip(mask_path,img_type,label_type))
# res=[]
# with Pool(150) as p:
#     res=p.map(extract_labeled_from_dcm_or_nii,compose_info)

# save={}
# for i in res:
#     save.update(i)
    
# save.pop('unvalid')
# data = json.dumps(save)
# f2 = open('new_json.json', 'w')
# f2.write(data)
# f2.close()      



def resample(image, scan, new_spacing=[1,1,1]):
    # Determine current pixel spacing
    spacing = np.array([scan[0].SliceThickness] + scan[0].PixelSpacing, dtype=np.float32)

    resize_factor = spacing / new_spacing
    new_real_shape = image.shape * resize_factor
    new_shape = np.round(new_real_shape)
    real_resize_factor = new_shape / image.shape
    new_spacing = spacing / real_resize_factor
    
    image = scipy.ndimage.interpolation.zoom(image, real_resize_factor, mode='nearest')
    
    return image, new_spacing

def plot_3d(image, threshold=-300):
    
    # Position the scan upright, 
    # so the head of the patient would be at the top facing the camera
    p = image.transpose(2,1,0)
    
    verts, faces = measure.marching_cubes(p, threshold)

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

def load_scan(filelist):#
    '''

    '''
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


def chunks(l, n):
    # Credit: Ned Batchelder
    # Link: http://stackoverflow.com/questions/312443/how-do-you-split-a-list-into-evenly-sized-chunks
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]

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

def chunk(list, n):
    result = []
    for i in range(n):
        result.append(list[math.floor(i / n * len(list)):math.floor((i + 1) / n * len(list))])
    return result
       
def run_multi_process(item_list, n_proc, func, with_proc_num=False):
    tasks = chunk(item_list, n_proc)
    if with_proc_num:
        for i in range(len(tasks)):
            tasks[i] = (i, tasks[i])
    with multiprocessing.Pool(processes=n_proc) as pool:
        results = pool.map(func, tasks)
    return results

def visual_nii():
    pass



def myshow(img, title=None, margin=0.05, dpi=80, cmap="gray"):
    nda = sitk.GetArrayFromImage(img)

    spacing = img.GetSpacing()
    slicer = False

    if nda.ndim == 3:
        # fastest dim, either component or x
        c = nda.shape[-1]

        # the the number of components is 3 or 4 consider it an RGB image
        if not c in (3, 4):
            slicer = True

    elif nda.ndim == 4:
        c = nda.shape[-1]

        if not c in (3, 4):
            raise RuntimeError("Unable to show 3D-vector Image")

        # take a z-slice
        slicer = True

    if slicer:
        ysize = nda.shape[1]
        xsize = nda.shape[2]
    else:
        ysize = nda.shape[0]
        xsize = nda.shape[1]

    # Make a figure big enough to accomodate an axis of xpixels by ypixels
    # as well as the ticklabels, etc...
    figsize = (1 + margin) * ysize / dpi, (1 + margin) * xsize / dpi

    def callback(z=None):

        extent = (0, xsize * spacing[1], ysize * spacing[0], 0)

        fig = plt.figure(figsize=figsize, dpi=dpi)

        # Make the axis the right size...
        ax = fig.add_axes([margin, margin, 1 - 2 * margin, 1 - 2 * margin])

        if z is None:
            ax.imshow(nda, extent=extent, interpolation=None, cmap=cmap)
        else:
            ax.imshow(nda[z, ...], extent=extent, interpolation=None, cmap=cmap)

        if title:
            plt.title(title)

        plt.show()

    if slicer:
        interact(callback, z=(0, nda.shape[0] - 1))
    else:
        callback()


def myshow3d(img, xslices=[], yslices=[], zslices=[], title=None, margin=0.05, dpi=80):
    size = img.GetSize()
    img_xslices = [img[s, :, :] for s in xslices]
    img_yslices = [img[:, s, :] for s in yslices]
    img_zslices = [img[:, :, s] for s in zslices]

    maxlen = max(len(img_xslices), len(img_yslices), len(img_zslices))

    img_null = sitk.Image([0, 0], img.GetPixelID(), img.GetNumberOfComponentsPerPixel())

    img_slices = []
    d = 0

    if len(img_xslices):
        img_slices += img_xslices + [img_null] * (maxlen - len(img_xslices))
        d += 1

    if len(img_yslices):
        img_slices += img_yslices + [img_null] * (maxlen - len(img_yslices))
        d += 1

    if len(img_zslices):
        img_slices += img_zslices + [img_null] * (maxlen - len(img_zslices))
        d += 1

    if maxlen != 0:
        if img.GetNumberOfComponentsPerPixel() == 1:
            img = sitk.Tile(img_slices, [maxlen, d])
        # TODO check in code to get Tile Filter working with VectorImages
        else:
            img_comps = []
            for i in range(0, img.GetNumberOfComponentsPerPixel()):
                img_slices_c = [sitk.VectorIndexSelectionCast(s, i) for s in img_slices]
                img_comps.append(sitk.Tile(img_slices_c, [maxlen, d]))
            img = sitk.Compose(img_comps)

    myshow(img, title, margin, dpi)
