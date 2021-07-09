#!/usr/bin/env python3
import numpy as np
import sys
import cv2
import model
import os

inputFileName= sys.argv[1]
(filename, _) = os.path.splitext(inputFileName)
fullname        = sys.argv[2]
threshold = sys.argv[3]
try:
    net = model.unet()
    net.load_weights(fullname)
except ValueError:
    print("Oops there are no model here")
img = cv2.imread(inputFileName, 0)
img=cv2.flip(img, 0)

#PREPARE DATA
h=img.shape[0]
hToPredict=img.shape[0]
w=img.shape[1]
wToPredict=img.shape[1]

#NORMALIZE
if h%model.numFilt != 0 or w%model.numFilt != 0:
    hToPredict = model.numFilt * round(img.shape[0] / model.numFilt)
    wToPredict = model.numFilt * round(img.shape[1] / model.numFilt)
    img = cv2.resize(img, (wToPredict,hToPredict), interpolation =cv2.INTER_AREA)
img = img / 255
img = np.reshape(img,img.shape+(1,))
img = np.reshape(img,(1,)+img.shape)
#    #PREDICTION
px 	= net.predict(img, verbose=1)
    #EXTRACT PROBABILITY
px	= px[0,:,:,0]
px=(px*255).astype(np.uint8)
if h!=hToPredict or w!=wToPredict :
    px = cv2.resize(px, (w,h), interpolation =cv2.INTER_AREA)
px=cv2.flip(px, 0)
cv2.imwrite(filename+'SEG.pgm', px )
#TRESHOLD
_,px_thresh = cv2.threshold(px,127,255,cv2.THRESH_BINARY)
#WRITE
cv2.imwrite(filename+'SEGTRESH.pgm', px_thresh )

    
