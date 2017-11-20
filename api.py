import torch
import torch.nn as nn
import os
from torch.autograd import Variable
from skimage import io, transform
import numpy as np
import cv2
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import re
from tensorboardX import SummaryWriter
import time
import sys
from Neural_Network_Class import conv_deconv

model=conv_deconv()
test = sys.argv[1]
img = cv2.imread(test)
oldHeight,oldWidth = img.shape[:2]
newWidth = 512
newHeight = 512
input_image = cv2.resize(img,(newWidth,newHeight))
input_image=torch.from_numpy(input_image.transpose((2, 0, 1)))
input_image=input_image.unsqueeze(0)
input_image=input_image.type(torch.FloatTensor)
input_image=Variable(input_image)
input_image=input_image.cuda()
model=torch.load("checkpoints/model_iter_11000.pt")
output_image=model(input_image)
output_image=output_image.squeeze()
output_image=output_image.cpu()
#output_image.data=output_image.data.type(torch.ByteTensor)
output_image=output_image.data.numpy()
output_image=output_image.transpose((1,2,0))
#r,g,b=cv2.split(output_image)
#output_image=cv2.merge((b,g,r))
# resizing
output_image =  cv2.resize(output_image,(oldWidth,oldHeight))

#display
cv2.imwrite('b.jpg',output_image)
os.system('eog b.jpg')
