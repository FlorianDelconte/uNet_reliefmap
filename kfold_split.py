########################################################################################
#This script is used to split patchs (320*320) in 5 folds                              #
########################################################################################

import sys, getopt
import os
from sklearn.model_selection import KFold
import glob
import cv2
labels_suffixe="labels"
examples_suffixe="exemples"
data_root="dbRelief/thumbnails/"
def lauch_data():
    pos_examples=[]
    pos_labels=[]
    neg_examples=[]
    neg_labels=[]
    for name in glob.glob(os.path.join(os.getcwd(), data_root,examples_suffixe,"*[!NEG].png")):
        pos_examples.append(name)
    for name in glob.glob(os.path.join(os.getcwd(), data_root,examples_suffixe,"*NEG.png")):
        neg_examples.append(name)
    for name in glob.glob(os.path.join(os.getcwd(), data_root,labels_suffixe,"*[!NEG].png")):
        pos_labels.append(name)
    for name in glob.glob(os.path.join(os.getcwd(), data_root,labels_suffixe,"*NEG.png")):
        neg_labels.append(name)

    return pos_examples,pos_labels,neg_examples,neg_labels


def main(argv):
    print("hello world")
    pos_examples,pos_labels,neg_examples,neg_labels=lauch_data()
    print("pos example size : "+str(len(pos_examples)))
    print("neg example size : "+str(len(neg_examples)))
    cv = KFold(n_splits=5, random_state=1, shuffle=True)
    k=0
    #create kfold directory

    if not os.path.exists(os.path.join(os.getcwd(), data_root, "kfold")):
        os.mkdir(os.path.join(os.getcwd(), data_root, "kfold"))

    for train_index, test_index in cv.split(pos_examples):
        print(test_index)
        os.mkdir(os.path.join(os.getcwd(), data_root, "kfold","k"+str(k)))
        os.mkdir(os.path.join(os.getcwd(), data_root, "kfold","k"+str(k),"train"))
        os.mkdir(os.path.join(os.getcwd(), data_root, "kfold","k"+str(k),"valid"))
        os.mkdir(os.path.join(os.getcwd(), data_root, "kfold","k"+str(k),"train","input"))
        os.mkdir(os.path.join(os.getcwd(), data_root, "kfold","k"+str(k),"train","output"))
        os.mkdir(os.path.join(os.getcwd(), data_root, "kfold","k"+str(k),"valid","input"))
        os.mkdir(os.path.join(os.getcwd(), data_root, "kfold","k"+str(k),"valid","output"))

        for i in train_index:
            img_example = cv2.imread(pos_examples[i], 0)
            cv2.imwrite(os.path.join(os.getcwd(), data_root, "kfold","k"+str(k),"train","input",str(i)+".png"),img_example)
            img_label = cv2.imread(pos_labels[i], 0)
            cv2.imwrite(os.path.join(os.getcwd(), data_root, "kfold","k"+str(k),"train","output",str(i)+".png"),img_label)
        for i in test_index:
            img_example = cv2.imread(pos_examples[i], 0)
            cv2.imwrite(os.path.join(os.getcwd(), data_root, "kfold","k"+str(k),"valid","input",str(i)+".png"),img_example)
            img_label = cv2.imread(pos_labels[i], 0)
            cv2.imwrite(os.path.join(os.getcwd(), data_root, "kfold","k"+str(k),"valid","output",str(i)+".png"),img_label)


        k=k+1
    k=0
    for train_index, test_index in cv.split(neg_examples):
        for i in train_index:
            #cv2.imwrite(os.getcwd()+data_root+"/kfold/"+"k_"+str(k)+"/train/input/"+str(i)+"neg.png", img_example_neg )
            #cv2.imwrite(os.getcwd()+data_root+"/kfold/"+"k_"+str(k)+"/train/output/"+str(i)+"neg.png", img_label_neg )
            img_example_neg = cv2.imread(neg_examples[i], 0)
            cv2.imwrite(os.path.join(os.getcwd(), data_root, "kfold","k"+str(k),"train","input",str(i)+"neg.png"),img_example_neg)
            img_label_neg = cv2.imread(neg_labels[i], 0)
            cv2.imwrite(os.path.join(os.getcwd(), data_root, "kfold","k"+str(k),"train","output",str(i)+"neg.png"),img_label_neg)
        for i in test_index:
            #cv2.imwrite(os.getcwd()+data_root+"/kfold/"+"k_"+str(k)+"/valid/input/"+str(i)+"neg.png", img_example_neg )
            #cv2.imwrite(os.getcwd()+data_root+"/kfold/"+"k_"+str(k)+"/valid/output/"+str(i)+"neg.png", img_label_neg )
            img_example_neg = cv2.imread(neg_examples[i], 0)
            cv2.imwrite(os.path.join(os.getcwd(), data_root, "kfold","k"+str(k),"valid","input",str(i)+"neg.png"),img_example_neg)
            img_label_neg = cv2.imread(neg_labels[i], 0)
            cv2.imwrite(os.path.join(os.getcwd(), data_root, "kfold","k"+str(k),"valid","output",str(i)+"neg.png"),img_label_neg)

        k=k+1


if __name__ == "__main__":
   main(sys.argv[1:])
