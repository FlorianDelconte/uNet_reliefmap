########################################################################################
#This script is use to extract 320*320 mini image around center                        #
#  of connected component in groundtruthimage                                          #
########################################################################################

#!/usr/bin/python3
import sys, getopt
import os
import cv2
import numpy as np
import random
#ratio of defect pixel in groundtruth to be acccepted like a negegative exemple
ratio_neg=0
#ratio for distribution in train dirs
ratio=70
#dimension of mini image
patchdim=320
#valid input path
validInputPath="valid/input/"
#valid output validInputPath
validOutputPath="valid/output/"
#train input path
trainInputPath="train/input/"
#train output path
trainOutputPath="train/output/"
def main(argv):
    imgPath = ''
    gtPath = ''
    outputDirectory = ''
    try:
        opts, args = getopt.getopt(argv,"hi:g:o:",["iImage=","IGT=","odir="])
    except getopt.GetoptError:
        print (' -i <inputImage> -g <inputGT> -o <outputDirectory>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print ('defect_detect.py -i <inputImage> -g <inputGT> -o <outputDirectory>')
            sys.exit()
        elif opt in ("-i", "--iImage"):
            imgPath = arg
        elif opt in ("-g", "--IGT"):
            gtPath = arg
        elif opt in ("-o", "--odir"):
            outputDirectory = arg

    print ('Path to image is ', imgPath)
    print ('Path to groundTruth is ', gtPath)
    print ('Output directory is ', outputDirectory)
    #cut img input in imagette (-patchdim-*-patchdim-) centered around barycentre of connected component
    posEx,posLab=cutPositiveExample(imgPath,gtPath)
    c = list(zip(posEx, posLab))
    random.shuffle(c)
    posEx, posLab = zip(*c)
    #inpaint to make negative exemple. No need to label for negative exepel (zeros matrix)
    negEx=cutNegativeExample(imgPath,gtPath)
    random.shuffle(negEx)
    #print(len(negEx))
    #########################################################################################################
    #                     DISTRIBUTE DATA IN TRAIN AND VALID DIRECTORY                                      #
    #########################################################################################################
    #extract image name to write file
    imageName=os.path.splitext(os.path.basename(imgPath))[0]
    #extract gt name to write file
    gtName=os.path.splitext(os.path.basename(gtPath))[0]
    #list of exemple and label need to have same size
    assert(len(posEx)==len(posLab))
    #compute how many example+label need to be distribute in training directory
    tresholdResPos=round((len(posEx)/100) * ratio)
    tresholdResNeg=round((len(negEx)/100) * ratio)
    print(str(ratio)+" % = "+str(tresholdResPos)+" imagettes in training directory")
    print(str(ratio)+" % = "+str(tresholdResNeg)+" imagettes in training directory")
    countpos=0
    countpos2=0
    #distribute positive exemple
    for i in range(len(posEx)):
        '''if i<=tresholdResPos :
            cv2.imwrite(outputDirectory+trainInputPath+imageName+"_"+str(i)+".png", posEx[i])
            cv2.imwrite(outputDirectory+trainOutputPath+gtName+"_"+str(i)+".png", posLab[i])
            countpos+=1
        else :
            cv2.imwrite(outputDirectory+validInputPath+imageName+"_"+str(i)+".png", posEx[i])
            cv2.imwrite(outputDirectory+validOutputPath+gtName+"_"+str(i)+".png",  posLab[i])
            countpos2+=1'''
        cv2.imwrite(outputDirectory+"/exemples/"+imageName+"_"+str(i)+".png", posEx[i])
        cv2.imwrite(outputDirectory+"/labels/"+imageName+"_"+str(i)+".png", posLab[i])
    #print("pos exemple in train : ",countpos )
    #print("pos exemple in valid : ",countpos2 )
    #distribute negative exemple
    count1=0
    count2=0
    for i in range(len(negEx)):
        '''if i<=tresholdResNeg :
            cv2.imwrite(outputDirectory+trainInputPath+imageName+"_"+str(i)+"NEG.png", negEx[i])
            cv2.imwrite(outputDirectory+trainOutputPath+gtName+"_"+str(i)+"NEG.png", np.zeros((patchdim, patchdim),np.uint8))
            count1+=1
        else :
            cv2.imwrite(outputDirectory+validInputPath+imageName+"_"+str(i)+"NEG.png", negEx[i])
            cv2.imwrite(outputDirectory+validOutputPath+gtName+"_"+str(i)+"NEG.png", np.zeros((patchdim, patchdim),np.uint8))
            count2+=1'''
        cv2.imwrite(outputDirectory+"/exemples/"+imageName+"_"+str(i)+"NEG.png", negEx[i])
        cv2.imwrite(outputDirectory+"/labels/"+imageName+"_"+str(i)+"NEG.png", np.zeros((patchdim, patchdim),np.uint8))
    #distribution in the same directory.

    #print("neg exemple in train : ",count1 )
    #print("neg exemple in valid : ",count2 )


def cutNegativeExample(imgPath,gtPath):
    #list of negative exemple
    negativeExample = []
    #list of negative label
    negativelabel = []
    #relief image
    image=cv2.imread(imgPath,0)
    #groundtruth image
    gt = cv2.imread(gtPath,0)
    #size gt and exemple
    h,w = gt.shape

    #extends gt and exemple (like a cylinder)
    gt_letfPart=gt[0:h,0:round(w/4)]
    gt_rightPart=gt[0:h,round(w/4):w]
    image_letfPart=image[0:h,0:round(w/4)]
    image_rightPart=image[0:h,round(w/4):w]
    gt_extend = cv2.hconcat([gt_rightPart,gt,gt_letfPart])
    image_extend = cv2.hconcat([image_rightPart,image,image_letfPart])
    h_ext,w_ext=gt_extend.shape

    #UNCOMMENT TO DISPLAY EXTENDS LABEL AND EXEMPLE
    #cv2.imshow('image',image_extend)
    #cv2.waitKey(0)
    '''cv2.imshow('image',gt_extend)
    cv2.waitKey(0)
    cv2.destroyAllWindows()'''
    counter1=0
    counter2=0
    for i in range(patchdim//2,h_ext-(patchdim//2),patchdim//2):
        for j in range(patchdim//2,w_ext-(patchdim//2),patchdim//2):
            print(gt_extend.shape)
            print(i,j)
            if gt_extend[i, j]!=255:
                #extract imagette
                negEx=gt_extend[i-(patchdim//2):i+(patchdim//2),j-(patchdim//2):j+(patchdim//2)]
                #compute raio of B/W
                current_ratio_BW=np.count_nonzero(negEx)/(patchdim*patchdim)

                #check ratio
                if(current_ratio_BW<=(ratio_neg/100)):
                    print(current_ratio_BW)
                    #extract imagette
                    currentNegLab=gt_extend[i-(patchdim//2):i+(patchdim//2),j-(patchdim//2):j+(patchdim//2)]
                    currentNegEx=image_extend[i-(patchdim//2):i+(patchdim//2),j-(patchdim//2):j+(patchdim//2)]
                    kernel = np.ones((10,10),np.uint8)
                    currentNegLab=cv2.dilate(currentNegLab,kernel,iterations = 1)
                    texture=cv2.bitwise_and(cv2.bitwise_not(currentNegLab), currentNegEx)
                    texture = cv2.inpaint(texture,currentNegLab,10,cv2.INPAINT_TELEA)
                    negativeExample.append(texture)
                    #UNCOMMENT TO DISPLAY PIPELINE ON LABEL + NEGATIVE FINAL EXEMPLE
                    '''cv2.imshow('image',currentNegLab)
                    cv2.waitKey(0)
                    cv2.imshow('image',cv2.dilate(currentNegLab,kernel,iterations = 1))
                    cv2.waitKey(0)
                    cv2.imshow('image',texture)
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()'''
                    counter1+=1
    print("image sur 1 : "+str(counter1))
    #print("image sur 2 : "+str(counter2))
    return negativeExample

def cutPositiveExampleNoIntersection(imgPath,gtPath):
    #list of negative exemple
    negativeExample = []
    #list of negative label
    negativelabel = []
    #relief image
    image=cv2.imread(imgPath,0)
    #groundtruth image
    gt = cv2.imread(gtPath,0)
    #size gt and exemple
    h,w = gt.shape

    #extends gt and exemple (like a cylinder)
    '''gt_letfPart=gt[0:h,0:round(w/4)]
    gt_rightPart=gt[0:h,round(w/4):w]
    image_letfPart=image[0:h,0:round(w/4)]
    image_rightPart=image[0:h,round(w/4):w]
    gt_extend = cv2.hconcat([gt_rightPart,gt,gt_letfPart])
    image_extend = cv2.hconcat([image_rightPart,image,image_letfPart])
    h_ext,w_ext=gt_extend.shape'''

    #UNCOMMENT TO DISPLAY EXTENDS LABEL AND EXEMPLE
    #cv2.imshow('image',image_extend)
    #cv2.waitKey(0)
    '''cv2.imshow('image',gt_extend)
    cv2.waitKey(0)
    cv2.destroyAllWindows()'''
    counter1=0
    counter2=0
    for i in range(patchdim//2,h_ext-(patchdim//2),patchdim//2):
        for j in range(patchdim//2,w_ext-(patchdim//2),patchdim//2):
            print(gt.shape)
            print(i,j)
            if gt[i, j]!=255:
                #extract imagette
                negEx=gt[i-(patchdim//2):i+(patchdim//2),j-(patchdim//2):j+(patchdim//2)]
                #compute raio of B/W
                current_ratio_BW=np.count_nonzero(negEx)/(patchdim*patchdim)

                #check ratio
                if(current_ratio_BW<=(ratio_neg/100)):
                    print(current_ratio_BW)
                    #extract imagette
                    currentNegLab=gt[i-(patchdim//2):i+(patchdim//2),j-(patchdim//2):j+(patchdim//2)]
                    currentNegEx=image[i-(patchdim//2):i+(patchdim//2),j-(patchdim//2):j+(patchdim//2)]
                    kernel = np.ones((10,10),np.uint8)
                    currentNegLab=cv2.dilate(currentNegLab,kernel,iterations = 1)
                    texture=cv2.bitwise_and(cv2.bitwise_not(currentNegLab), currentNegEx)
                    texture = cv2.inpaint(texture,currentNegLab,10,cv2.INPAINT_TELEA)
                    negativeExample.append(texture)
                    #UNCOMMENT TO DISPLAY PIPELINE ON LABEL + NEGATIVE FINAL EXEMPLE
                    '''cv2.imshow('image',currentNegLab)
                    cv2.waitKey(0)
                    cv2.imshow('image',cv2.dilate(currentNegLab,kernel,iterations = 1))
                    cv2.waitKey(0)
                    cv2.imshow('image',texture)
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()'''
                    counter1+=1
    print("image sur 1 : "+str(counter1))
    #print("image sur 2 : "+str(counter2))
    return negativeExample

def cutPositiveExample(imgPath,gtPath):
    #list of positive exemple
    positiveExample = []
    #list of positive label
    positivelabel = []
    #relief image
    image=cv2.imread(imgPath,0)
    #groundtruth image
    gt = cv2.imread(gtPath,0)
    #find center of connected component
    centroids=find_connected_component(gt)
    for i in range(1,len(centroids)):
        print("nb imagettes : "+str(len(centroids)))
        #find top left corner of mini image
        tlc=getTopLeftCorner_Patch(gt,centroids[i])
        #create mini image
        mini_gt=gt[tlc[1]:tlc[1]+patchdim,tlc[0]:tlc[0]+patchdim]
        mini_img=image[tlc[1]:tlc[1]+patchdim,tlc[0]:tlc[0]+patchdim]
        positiveExample.append(mini_img)
        positivelabel.append(mini_gt)
    return positiveExample,positivelabel

def getTopLeftCorner_Patch(dtimg,centre):
    ct=0
    topLeftCorner=[int(centre[0]-patchdim/2),int(centre[1]-patchdim/2)]
    botRightCorner=[int(centre[0]+patchdim/2),int(centre[1]+patchdim/2)]
    ymax=dtimg.shape[0]
    xmax=dtimg.shape[1]


    print(str(xmax)+" * "+str(ymax))

    corner_output=[-1,-1]
    #tous les cas possible :
    #1.2.3
    #4.5.6
    #7.8.9
    #cas n°1
    if topLeftCorner[0]<0 and topLeftCorner[1]<0 :
        corner_output[0]=0
        corner_output[1]=0
        ct=ct+1
        print("cas n°1")
    #cas n°2
    if topLeftCorner[0]>=0 and topLeftCorner[1]<0 and botRightCorner[0]<=xmax and botRightCorner[1]<=ymax :
        corner_output[0]=topLeftCorner[0]
        corner_output[1]=0
        ct=ct+1
        print("cas n°2")
    #cas n°3
    if topLeftCorner[1]<0 and botRightCorner[0]>xmax:
        corner_output[0]=topLeftCorner[0]-abs(botRightCorner[0]-xmax)
        corner_output[1]=0
        ct=ct+1
        print("cas n°3")
    #cas n°4
    if topLeftCorner[0]<0 and topLeftCorner[1]>=0 and botRightCorner[0]<=xmax and botRightCorner[1]<=ymax:
        corner_output[0]=0
        corner_output[1]=topLeftCorner[1]
        ct=ct+1
        print("cas n°4")
    #cas n°5
    if topLeftCorner[0]>=0 and topLeftCorner[1]>=0 and botRightCorner[0]<=xmax and botRightCorner[1]<=ymax :
        corner_output=topLeftCorner
        ct=ct+1
        print("cas n°5")
    #cas n°6
    if topLeftCorner[0]>=0 and topLeftCorner[1]>=0 and botRightCorner[0]>xmax and botRightCorner[1]<=ymax :
        corner_output[0]=topLeftCorner[0]-abs(botRightCorner[0]-xmax)
        corner_output[1]=topLeftCorner[1]
        ct=ct+1
        print("cas n°6")
    #cas n°7
    if topLeftCorner[0]<0 and botRightCorner[1]>ymax :
        corner_output[0]=0
        corner_output[1]=topLeftCorner[1]-abs(botRightCorner[1]-ymax)
        ct=ct+1
        print("cas n°7")
    #cas n°8
    if topLeftCorner[0]>=0 and topLeftCorner[1]>=0 and botRightCorner[1]>ymax and botRightCorner[0]<= xmax:
        corner_output[0]=topLeftCorner[0]
        corner_output[1]=topLeftCorner[1]-abs(botRightCorner[1]-ymax)
        ct=ct+1
        print("cas n°8")
    #cas n°9
    if botRightCorner[0]>xmax and botRightCorner[1]>ymax:
        corner_output[0]=topLeftCorner[0]-abs(botRightCorner[0]-xmax)
        corner_output[1]=topLeftCorner[1]-abs(botRightCorner[1]-ymax)
        ct=ct+1
        print("cas n°9")
    print("centre :"+str(centre))
    print("top left before : "+ str(topLeftCorner))
    print("bot right before : "+ str(botRightCorner))

    print("topleftcorner after :"+str(corner_output))
    print("count : "+str(ct))
    return corner_output
def find_connected_component(gt):
    # need to choose 4 or 8 for connectivity type
    connectivity = 8
    # find connected component
    output = cv2.connectedComponentsWithStats(gt, connectivity, cv2.CV_32S)
    # The first cell is the number of labels
    num_labels = output[0]
    # The second cell is the label matrix
    labels = output[1]
    # The third cell is the stat matrix
    stats = output[2]
    # The fourth cell is the centroid matrix
    centroids = output[3]
    print ("number of connected component : "+str(num_labels))
    return centroids






if __name__ == "__main__":
   main(sys.argv[1:])
