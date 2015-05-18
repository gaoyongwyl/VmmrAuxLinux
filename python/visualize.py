#!/usr/bin/python2.7
#-*-coding:utf-8-*-

#import pdb

import caffe
import numpy as np
import matplotlib.pyplot as plt
from caffe.proto import caffe_pb2
from caffe import io
import VmmrAuxiliary as Va
import time

caffe_root = '/home/ygao/Projects/VehicleRecogntition/Code/caffe/'

import os
import sys

sys.path.insert(0, os.path.join('python'))

plt.rcParams['figure.figsize'] = (10,10)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'


# take an array of shape (n, height, width) or (n, height, width, channels)
# and visualize each (height, width) thing in a grid of size approx. sqrt(n) by sqrt(n)
def vis_square(data, padsize=1, padval=0):
    data -= data.min()
    data /= data.max()
    
    # force the number of filters to be square
    n = int(np.ceil(np.sqrt(data.shape[0])))
    padding = ((0, n ** 2 - data.shape[0]), (0, padsize), (0, padsize)) + ((0, 0),) * (data.ndim - 3)
    data = np.pad(data, padding, mode='constant', constant_values=(padval, padval))
    
    # tile the filters into an image
    data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
    data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])
    
    plt.imshow(data)


#caffe.set_mode_cpu()
#net = caffe.Net(caffe_root + 'models/bvlc_reference_caffenet/deploy.prototxt',
#                   caffe_root + 'models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel',
#                   caffe.TEST)

NetPrototxtFile = '/home/ygao/Projects/VehicleRecogntition/DNN_Experiments/V1_AAuMColor/Makemodel_C_vface_150/Exp_20150515_03H21M_04s/vmakemodel_C_vface_150_deploy.prototxt'
NetBinaryModel = '/home/ygao/Projects/VehicleRecogntition/DNN_Experiments/V1_AAuMColor/Makemodel_C_vface_150/Exp_20150515_03H21M_04s/vmakemodel_C_vface_150_iter_150000.caffemodel'
MeanFile = '/home/ygao/Projects/VehicleRecogntition/Data/V1_AAuMColor/Caffe_Cropped_C_vface_150/vmakemodel_C_vface_150_image_mean.mean'
ExampleImage = '/home/ygao/Projects/VehicleRecogntition/Data/V1_AAuMColor/Cropped_C_vface_150/中国重汽_F/中国重汽_F_00069360-00_A0.jpg'

MakemodelLabelDictFile = '/home/ygao/Projects/VehicleRecogntition/Data/V1_AAuMColor/_LabelList_SL/Makemodel/Makemodel_ClassLabelDict.txt'

LabelClassNameDict = Va.ReadLabelClassNameDict(MakemodelLabelDictFile ) 


caffe.set_mode_cpu()
net = caffe.Net( NetPrototxtFile, NetBinaryModel, caffe.TEST )

data_shape = net.blobs['data'].data.shape
print( 'Data layer shape: {0}'.format(data_shape ) )

# input preprocessing: 'data' is the name of the input blob == net.inputs[0]
meanfd = open( MeanFile, 'rb')
meanblob = caffe_pb2.BlobProto()
meanblob.MergeFromString( meanfd.read() )
meandata= io.blobproto_to_array(meanblob)[0]
print( 'mean data shape is {0}'.format( meandata.shape ) )

transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_transpose('data', (2,0,1))
transformer.set_mean('data', meandata) # mean pixel
#transformer.set_raw_scale('data',  0.00390625)  # the reference model operates on images in [0,255] range instead of [0,1]. python represent image in [0,1] already
transformer.set_raw_scale('data',  255)
transformer.set_channel_swap('data', (2,1,0))  # the reference model has channels in BGR order instead of RGB

net.blobs['data'].reshape(1,3, data_shape[-2], data_shape[-1] )
t_data = transformer.preprocess('data', caffe.io.load_image(ExampleImage))
net.blobs['data'].data[...] = t_data
print( "t_data shape: {0}".format(t_data.shape) )
#plt.imshow(transformer.deprocess('data', net.blobs['data'].data[0]))
#plt.show()
#os.system( "read -n 1" )

out = net.forward()

predict_label = out['prob'].argmax()
print("Predicted class is # {0}.".format(out['prob'].argmax()))
print("Predicted class name is {0}.".format( LabelClassNameDict[ predict_label] ) )

#layerNameShapeList = [(k, v.data.shape) for k, v in net.blobs.items() ]
#for layername, shape in layerNameShapeList:
#    print( layername, shape )

#plt.imshow(  caffe.io.load_image(ExampleImage) )
plt.imshow(transformer.deprocess('data', net.blobs['data'].data[0]))

# the parameters are a list of [weights, biases]
filters = net.params['conv0'][0].data
vis_square(filters.transpose(0, 2, 3, 1))
plt.show()

#The first layer output, conv1 (rectified responses of the filters above, first 36 only)
feat = net.blobs['conv0'].data[0, :32]
vis_square(feat, padval=1)
plt.show()


#The first fully connected layer, fc6 (rectified)
feat = net.blobs['ip1'].data[0]
plt.subplot(2, 1, 1)
plt.plot(feat.flat)
plt.subplot(2, 1, 2)
_ = plt.hist(feat.flat[feat.flat > 0], bins=100)
plt.show()

#We show the output values and the histogram of the positive values
feat = net.blobs['ip2'].data[0]
plt.subplot(2, 1, 1)
plt.plot(feat.flat)
plt.subplot(2, 1, 2)
_ = plt.hist(feat.flat[feat.flat > 0], bins=100)
plt.show()

#The final probability output, prob
feat = net.blobs['prob'].data[0]
plt.plot(feat.flat)
plt.show()

