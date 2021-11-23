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
GRID_SIZE=320
shift=GRID_SIZE//2
#dUnlarge image
unlargedImg =  np.zeros([h+GRID_SIZE*2,w+GRID_SIZE*2],np.uint8)
unlargedImg[GRID_SIZE:h+GRID_SIZE,GRID_SIZE:w+GRID_SIZE]=img
#SPLIT INTO TILES
tiles = []
y=0
x=0
while y <= unlargedImg.shape[0] - GRID_SIZE :
    while x <= unlargedImg.shape[1] - GRID_SIZE :
        roi = unlargedImg[y:y + GRID_SIZE ,x:x + GRID_SIZE]
        tiles.append(roi)
        x+=shift
    y+=shift
    x=0
#PREDICT ON TILES
tilesResult = []
for e in tiles:
    #NORMALIZE
    e = e / 255
    #SHAPE FOR TENSOR
    e = np.reshape(e,e.shape+(1,))
    e = np.reshape(e,(1,)+e.shape)
    px 	= net.predict(e, verbose=1)
    px = px[0,:,:,0]
    tilesResult.append(px)
#CONCAT TILES
concatenedResult = np.zeros((unlargedImg.shape[0], unlargedImg.shape[1]), dtype = "float64")
#tiles id
i=0
y=0
x=0
while i < len(tilesResult):
    concatenedResult[y:y + shift,x:x + shift]=tilesResult[i][shift//2:shift+shift//2,shift//2:shift+shift//2]
    i+=1
    x+=shift
    if(x>=unlargedImg.shape[1] - GRID_SIZE + 1 ):
        x=0
        y+=shift

concatenedResult=(concatenedResult*255).astype(np.uint8)
#EXTRACT prediction from concat tile result
pred=concatenedResult[GRID_SIZE-(shift//2):h+GRID_SIZE-(shift//2),GRID_SIZE-(shift//2):w+GRID_SIZE-(shift//2)]
pred=cv2.flip(pred, 0)
#TRESHOLD
_,px_thresh = cv2.threshold(pred,127,255,cv2.THRESH_BINARY)
#WRITE IN PNG FOR IPOL DISPLAY
cv2.imwrite('outputSEGTRESH.png', px_thresh )
cv2.imwrite('outputSEG.png', pred )
#WRITE IN PGM TO CALL "Main_segToMesh.cpp"
cv2.imwrite(filename+'SEGTRESH.pgm', px_thresh )
print("PREDICTION FINISH")
