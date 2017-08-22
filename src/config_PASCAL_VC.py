import cv2,os,glob,pickle,sys,math,time
import numpy as np
from myresize import myresize

all_categories=['aeroplane','bicycle', 'boat', 'bottle', 'bus', 'car', 'chair', 'diningtable', 'motorbike', 'sofa', 'train', 'tvmonitor']

net_type='VGG'
# Alexnet
if net_type=='alex':
    Apad_set = [0, 0, 16, 16, 32, 48, 64] # padding size
    Astride_set = [4, 8, 8, 16, 16, 16, 16] # stride size
    featDim_set = [96, 96, 256, 256, 384, 384, 256] # feature dimension
    Arf_set = [11, 19, 51, 67, 99, 131, 163]
    offset_set = np.ceil(np.array(Apad_set)/np.array(Astride_set)).astype(int)
    layer_n = 4 # conv3
elif net_type=='VGG':
    Apad_set = [2, 6, 18, 42, 90] # padding size
    Astride_set = [2, 4, 8, 16, 32] # stride size
    featDim_set = [64, 128, 256, 512, 512] # feature dimension
    Arf_set = [6, 16, 44, 100, 212]
    offset_set = np.ceil(np.array(Apad_set).astype(float)/np.array(Astride_set)).astype(int)
    layer_n = 3 # pool4
    # layer_n = 2 # pool3
    # layer_n = 1 # pool2
    
Apad = Apad_set[layer_n]
Astride = Astride_set[layer_n]
featDim = featDim_set[layer_n]
Arf = Arf_set[layer_n]
offset = offset_set[layer_n]

scale_size = 224


Dataset = dict()
Dataset['img_dir_org'] = '/export/home/qliu24/dataset/PASCAL3D+_release1.1/Images/{0}_imagenet/'
Dataset['img_dir'] = '/export/home/qliu24/dataset/PASCAL3D+_cropped/{0}_imagenet/'
Dataset['anno_dir'] = '/export/home/qliu24/dataset/PASCAL3D+_release1.1/Annotations/{0}_imagenet/'
Dataset['list_dir'] = '/export/home/qliu24/dataset/PASCAL3D+_release1.1/Image_sets'
Dataset['train_list'] = os.path.join(Dataset['list_dir'], '{}_imagenet_train.txt')
Dataset['test_list'] = os.path.join(Dataset['list_dir'], '{}_imagenet_val.txt')


VC = dict()
VC['num'] = 512

if net_type=='alex':
    VC['layer'] = 'conv3'
elif net_type=='VGG':
    VC['layer'] = 'pool4'
    # VC['layer'] = 'pool3'
    # VC['layer'] = 'pool2'

model_cache_folder = '/export/home/qliu24/qing_voting_139/qing_voting_py/cache/'

root_dir = '/export/home/qliu24/PASCAL_VC/'

Dict = dict()
Dict['cache_path'] = os.path.join(root_dir, 'feat', '{0}_all_dumped_data'.format(VC['layer']))
Dict['Dictionary'] = os.path.join(root_dir, 'dictionary', 'dictionary_PASCAL3D+_VGG16_{}_K{}_vMFMM30.pickle'.format(VC['layer'], VC['num']))

Feat = dict()
Feat['cache_dir'] = os.path.join(root_dir, 'feat')
# Feat['cache_dir_val'] = os.path.join(root_dir, 'feat', 'val')
if not os.path.exists(Feat['cache_dir']):
    os.makedirs(Feat['cache_dir'])
    
Model_dir = os.path.join(root_dir, 'mix_model')
Result_dir = os.path.join(root_dir, 'result')