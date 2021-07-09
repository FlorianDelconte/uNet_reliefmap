import numpy as np
import os
import sys
import skimage.io as io
import model
import cv2
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from skimage.transform import resize
from skimage import img_as_bool
import csv
import matplotlib.pyplot as plt
import re

prefix_name_model="k"
path_model = os.path.join(os.getcwd(),'model', 'save','repartition-VT','k-fold_ICPR2020')
path_write_data= path_model

def main(argv):
    nb_fold=5
    model_results = []
    #path to find data test

    #TO TEST K-fold
    for i in range(nb_fold):
        path_test_input     = os.path.join(os.getcwd(),'..','crossValidationICPR2020','data_'+str(i), 'test', 'input')
        path_test_gt        = os.path.join(os.getcwd(),'..','crossValidationICPR2020','data_'+str(i), 'test', 'gt')
        fullpathModel       ="".join([path_model, '/', prefix_name_model,str(i),'.hdf5'])
        t_images,t_labels,t_fileName=prepare_data_testFullSize(path_test_input,path_test_gt)
        model_results.append(measuresFoldFullSize(t_images,t_labels,fullpathModel))

    writeMeasures(model_results)
    #TO TEST normal
    #path_test_input     = os.path.join(os.getcwd(),'..','repartitionParamVT','test', 'input')
    #path_test_gt        = os.path.join(os.getcwd(),'..','repartitionParamVT','test', 'gt')

    #fullpathModel       ="".join([path_model, '/', prefix_name_model,'.hdf5'])
    #t_images,t_labels,t_fileName=prepare_data_testFullSize(path_test_input,path_test_gt)
    #model_results.append(measuresFoldFullSize(t_images,t_labels,fullpathModel))

    #writeMeasures(model_results)
    #img_gen,label_gen=prepare_data_testFullSizev2( os.path.join(os.getcwd(),'..','repartitionData9','test'))
    #flow_gen=prepare_data_testFullSizev2( os.path.join(os.getcwd(),'..','repartitionData9','test'))
    #fullpathModel       ="".join([path_model, '/', prefix_name_model,'.hdf5'])
    #measuresFoldFullSizev2(img_gen,label_gen,fullpathModel)
    #measuresFoldFullSizev2(img_gen,label_gen,fullpathModel)
def writeMeasures(mod_res):
    nbmeasure=len(mod_res[0][0])
    #convert list to array
    np_mod_res = np.asarray(mod_res,float)

    #nb line in array
    h=np_mod_res.shape[0]
    #create numpy array to store mean of all K-fold

    mean_all_fold = np.empty([0,nbmeasure])
    #row titles
    #row_titles=np.empty(h, dtype='|S5')
    #CSV WRITE DATA PER FOLD
    for i in range(h):
        current_fold_measures=np_mod_res[i]



        mean_fold=np.mean(current_fold_measures, axis = 0)
        mean_all_fold = np.vstack([mean_all_fold, mean_fold])

        f_fold = path_write_data+"/"+prefix_name_model+"/"+prefix_name_model+str(i)
        #create a directory if doesn't exist
        if not os.path.exists(os.path.dirname(f_fold)):
            try:
                os.makedirs(os.path.dirname(f_fold))
            except OSError as exc: # Guard against race condition
                if exc.errno != errno.EEXIST:
                    raise
        #current_fold_measures = np.append(current_fold_measures, test_fileName, axis=1)
        current_nb_exemple=len(np_mod_res[i])
        rows = np.array(range(1,current_nb_exemple+1),dtype=int)[:, np.newaxis]
        #print(np.hstack((rows, current_fold_measures)))
        with open(f_fold, 'w') as f:
            np.savetxt(f,  np.hstack((rows, current_fold_measures)), delimiter=',',fmt=['%i','%f','%f','%f','%f'],header='loss,precision,recall,F1')
            f.write("\n")
            np.savetxt(f, mean_fold, delimiter=',',fmt='%f',newline=" ")
    #CSV WRITE MEAN ON ALL MEASURE
    #print(row_titles)
    mean_Kfold=np.mean(mean_all_fold, axis = 0)
    std_Kfold=np.std(mean_all_fold, axis = 0)
    f_model = path_write_data+"/"+prefix_name_model+"_measures.txt"
    with open(f_model, 'w') as f:
        np.savetxt(f, mean_all_fold, delimiter=',',fmt='%s',header='M-loss,M-precision,M-recall,M-F1')
        f.write("\n")
        np.savetxt(f, mean_Kfold, delimiter=',',fmt='%f',newline=" ")
        f.write("\n")
        np.savetxt(f, std_Kfold, delimiter=',',fmt='%f',newline=" ")
def sorted_alphanumeric(data):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ]
    return sorted(data, key=alphanum_key)
#function to prepare image to test on model architecture
def prepare_data_testFullSize(path_test_input,path_test_gt):
    test_images = []
    test_labels = []
    test_fileName = []
    #TEST IMAGES
    for image_name in sorted_alphanumeric(os.listdir(path_test_input)):
        print("IMAGES PATH : "+path_test_input+image_name)

        image_full_path = "{}/{}".format(path_test_input, image_name)
        img = io.imread(image_full_path, as_gray = True)

        h=img.shape[0]
        w=img.shape[1]

        if h%model.numFilt != 0 or w%model.numFilt != 0:
            print("Basic size : ("+str(h)+","+str(w)+")")
            print("Image input dosn't not match model shape... Resize in nearest multiple of "+str(model.numFilt))
            hToPredict = model.numFilt * round(img.shape[0] / model.numFilt)
            wToPredict = model.numFilt * round(img.shape[1] / model.numFilt)
            img = cv2.resize(img, (wToPredict,hToPredict), interpolation =cv2.INTER_AREA)
            print("New size : ("+str(hToPredict)+","+str(wToPredict)+")")

        img = img / 255.
        img = np.reshape(img,img.shape+(1,))
        img = np.reshape(img,(1,)+img.shape)
        print(img.shape)
        test_images.append(img)
        name, file_extension = os.path.splitext(image_name)
        test_fileName.append(name)
    #LABLES IMAGES
    for image_name in sorted_alphanumeric(os.listdir(path_test_gt)):

        print("LABELS PATH : "+path_test_gt+image_name)
        image_full_path = "{}/{}".format(path_test_gt, image_name)
        img = io.imread(image_full_path, as_gray = True)
        h=img.shape[0]
        w=img.shape[1]
        if h%model.numFilt != 0 or w%model.numFilt != 0:
            hToPredict = model.numFilt * round(img.shape[0] / model.numFilt)
            wToPredict = model.numFilt * round(img.shape[1] / model.numFilt)
            img=img_as_bool(resize(img, (hToPredict, wToPredict)))

        img = np.reshape(img,img.shape+(1,))
        img = np.reshape(img,(1,)+img.shape)
        print(img.shape)
        test_labels.append(img)


    return test_images,test_labels,test_fileName

def prepare_data_testFullSizev2(path_test):
    print(path_test)


    image_datagen = ImageDataGenerator(rescale=1./255)

    '''gen_flow = image_datagen.flow_from_directory(path_test,
                                                classes = ['gt','input'],
                                                class_mode=None,
                                                target_size=(320, 320),
                                                 shuffle=False)'''
    image_flow = image_datagen.flow_from_directory(path_test,classes = ['inputcut'],class_mode='categorical',color_mode='grayscale',target_size=(320,320), shuffle=False,batch_size=1,seed=1)
    masque_flow = image_datagen.flow_from_directory(path_test,classes = ['gtcut'],class_mode='categorical',color_mode='grayscale',target_size=(320,320),shuffle=False,batch_size=1,seed=1)
    print(image_flow.filenames)
    print(masque_flow.filenames)
    return image_flow,masque_flow
    #return gen_flow
def my_image_mask_generator(image_data_generator, mask_data_generator):
    while True:
        x = image_data_generator.next()
        y = mask_data_generator.next()
        yield (x[0], y[0])


def measuresFoldFullSize(t_images,t_labels,fullpathModel):
    k_actu_results=[]

    net = model.unet()
    net.load_weights(fullpathModel)
    #RESIZE label into IMAGE size
    for t, l in zip(t_images, t_labels):
        '''la=l
        ex = t
        ex	= ex[0,:,:,0]
        la =la[0,:,:,0]
        ex=(ex*255).astype(np.uint8)
        la=(la*255).astype(np.uint8)
        imgplot = plt.imshow(ex)
        plt.show()
        imgplot = plt.imshow(la)
        plt.show()'''
        res = net.evaluate(t,l,batch_size=1)

        k_actu_results.append(res)
        '''px 	= net.predict(t,verbose=1)
        px	= px[0,:,:,0]
        px=(px*255).astype(np.uint8)'''
        #imgplot = plt.imshow(px)
        #plt.show()

    return k_actu_results

def measuresFoldFullSizev2(img_gen,label_gen,fullpathModel):
    k_actu_results=[]

    '''x = img_gen.next()
    y = label_gen.next()
    image = x[0]
    imgplot = plt.imshow(image)
    plt.show()
    image = y[0]
    imgplot = plt.imshow(image)
    plt.show()'''
    #my_generator=my_image_mask_generator(img_gen,label_gen)



    net = model.unet()

    net.load_weights(fullpathModel)
    #print("BBBBBBBBBBBBBBBBBBOOOOOOOOOOOOOOOOONJJJJJJJJJJJJJJJJOUUUUUUUUUUUUUUUUUUUUUURRRRRRRRRRRRRRRRRRRRRRRRR")
    #x,y = flow_gen.next()
    #plt.show(x)
    net.evaluate(my_generator)
    #net.predict(my_generator)
    #net.fit(my_generator)
    #RESIZE label into IMAGE size







if __name__ == "__main__":
   main(sys.argv[1:])
