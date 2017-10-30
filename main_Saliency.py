
# Saliency Detection by Forward and Backward Cues in Deep-CNN, IEEE ICIP 2017
# REvised code for the top-down saliency computation part
import chainer
import numpy as np
import cv2 as cv
import cPickle as pickle
from PIL import Image
import os

from chainer import cuda
from chainer import serializers
from chainer import Variable
import chainer.functions as F

from VGGNet import VGGNet
from VGGNet_Input_Layer2Out import VGGNet_Input_Layer2Out
from VGGNetLayer3 import VGGNetLayer3
from VGGNetLayer4 import VGGNetLayer4
from VGGNetLayer5 import VGGNetLayer5
from VGGNetLayerFC import VGGNetLayerFC


imagefolder = "images"
resultfolder = "."

mean = np.array([103.939, 116.779, 123.68])

vggall = VGGNet()
serializers.load_hdf5('VGG.model', vggall)

# Submodel from input layer to layer 2 output
vgg2 = VGGNet_Input_Layer2Out()
vgg2.conv1_1 = vggall.conv1_1.copy() 
vgg2.conv1_2 = vggall.conv1_2.copy()
vgg2.conv2_1 = vggall.conv2_1.copy() 
vgg2.conv2_2 = vggall.conv2_2.copy()

# Submodel for layer 3
vgg3 = VGGNetLayer3()
vgg3.conv3_1 = vggall.conv3_1.copy() 
vgg3.conv3_2 = vggall.conv3_2.copy()
vgg3.conv3_3 = vggall.conv3_3.copy() 

# Submodel for layer 4
vgg4 = VGGNetLayer4()
vgg4.conv4_1 = vggall.conv4_1.copy() 
vgg4.conv4_2 = vggall.conv4_2.copy()
vgg4.conv4_3 = vggall.conv4_3.copy() 

# Submodel for layer 5
vgg5 = VGGNetLayer5()
vgg5.conv5_1 = vggall.conv5_1.copy() 
vgg5.conv5_2 = vggall.conv5_2.copy()
vgg5.conv5_3 = vggall.conv5_3.copy()

# Submodel for layer fcl
vggFC = VGGNetLayerFC()
vggFC.fc6 = vggall.fc6.copy()
vggFC.fc7 = vggall.fc7.copy()
vggFC.fc8 = vggall.fc8.copy()

included_extenstions = ['jpg', 'bmp', 'png', 'gif']
file_names = [fn for fn in os.listdir(imagefolder)
              if any(fn.endswith(ext) for ext in included_extenstions)]

radius = 112
y,x = np.ogrid[-radius:radius, -radius:radius]
mask =  np.sqrt(np.float32(x**2 + y**2))
mask = mask/mask.max()
mask = 0.75*np.abs(mask-1)+0.25

total_time = 0
for item in file_names:
	print item
	
	img_org = cv.imread(imagefolder+"/"+item)	
	img = np.float32(img_org)
	img -= mean
	img = cv.resize(img, (224, 224)).transpose((2, 0, 1))
	img = img[np.newaxis, :, :, :]

	imgv = Variable(img)

	# Calculate forward pass features from sub-modules of VGG16	
	h2 = vgg2(imgv)
	h3 = vgg3(h2)
	h4 = vgg4(h3)
	h5 = vgg5(h4)

	# calculate BP at layer fcl
	h = Variable(h5.data)
	hfc = vggFC(h)
	hfc.grad = hfc.data.copy()
	hfc.backward()
	h.grad[h.grad<0] = 0
	bpfc = h.grad.copy()
			
	# calculate BP at layer 5
	h = Variable(h4.data)
	h5 = vgg5(h)
	h5.grad = bpfc.copy()
	h5.backward()
	h.grad[h.grad<0] = 0
	bpl5 = h.grad.copy()

	# calculate BP at layer 4
	h = Variable(h3.data)
	h4 = vgg4(h)
	h4.grad = bpl5.copy()
	h4.backward()
	h.grad[h.grad<0] = 0
	bpl4 = h.grad.copy()

	# calculate BP at layer 3
	h = Variable(h2.data)
	h3 = vgg3(h)
	h3.grad = bpl4.copy()
	h3.backward()
	h.grad[h.grad<0] = 0
	bpl3 = h.grad.copy()
	
	filterSize = 11;
	size_img_org = img_org.shape
	
	lfc = cv.resize((h5.data*bpfc)[0].transpose((1,2,0)),(224,224))
	l5  = cv.resize((h4.data*bpl5)[0].transpose((1,2,0)),(224,224))
	l4  = cv.resize((h3.data*bpl4)[0].transpose((1,2,0)),(224,224))
	l3  = cv.resize((h2.data*bpl3)[0].transpose((1,2,0)),(224,224))
	
	saliencyFWBW = mask*(0.1*l3.sum(axis=2)+0.5*l4.sum(axis=2)+1*l5.sum(axis=2)+0.1*lfc.sum(axis=2))
	saliencyFWBW  = cv.GaussianBlur(saliencyFWBW,(filterSize,filterSize),0);	
	saliencyFWBW = cv.resize(saliencyFWBW,(size_img_org[1],size_img_org[0]))
				
	saliency = saliencyFWBW
	saliency -= saliency.min()
	saliency  = saliency/saliency.max()
	saliency  = np.uint8( np.round( 255*saliency ) )
	
	result = Image.fromarray((saliency).astype(np.uint8))
	result.save(resultfolder+"/"+item)
