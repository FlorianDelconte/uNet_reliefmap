from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import os
import glob
#import skimage.io as io
import model
import tensorflow as tf
def trainset_generator(batch_size,
                       train_path,
                       image_folder,
                       mask_folder,
                       aug_dict,
                       image_color_mode = "grayscale",
                       mask_color_mode  = "grayscale",
                       image_save_prefix= "image",
                       mask_save_prefix = "mask",
                       flag_multi_class = False,
                       num_class = 2,
                       save_to_dir =None,
                       target_size = (model.height, model.width),
                       seed = 1):
    '''
        can generate image and mask at the same time
        use the same seed for image_datagen and mask_datagen to ensure the transformation for image and mask is the same
        if you want to visualize the results of generator, set save_to_dir = "your path"
        '''
    image_datagen   = ImageDataGenerator(**aug_dict)
    mask_datagen    = ImageDataGenerator(**aug_dict)
    valid_datagen   = ImageDataGenerator(**aug_dict)

    image_generator = image_datagen.flow_from_directory(
                                                    train_path,
                                                    classes = [image_folder],
                                                    class_mode = 'categorical',
                                                    target_size = target_size,
                                                    color_mode=image_color_mode,
                                                    batch_size = batch_size,
                                                    save_to_dir = save_to_dir,
                                                    save_prefix  = image_save_prefix,
                                                    seed = seed)
    mask_generator = mask_datagen.flow_from_directory(
                                                  train_path,
                                                  classes = [mask_folder],
                                                  class_mode = 'categorical',
                                                  target_size = target_size,
                                                  color_mode = mask_color_mode,
                                                  batch_size = batch_size,
                                                  save_to_dir = save_to_dir,
                                                  save_prefix  = mask_save_prefix,
                                                  seed = seed)

    while True:

        x = image_generator.next()
        y = mask_generator.next()
        #print("generator1 : "+str(x[0].shape),str(y[0].shape))
        yield (x[0], y[0])
def trainset_generator2(data,
                        batch_size,
                       aug_dict,
                       image_save_prefix= "image",
                       mask_save_prefix = "mask",
                       save_to_dir =None,
                       seed = 1):

    image_datagen = ImageDataGenerator(**aug_dict)
    mask_datagen = ImageDataGenerator(**aug_dict)
    examples=data[0]
    labels = data[1]
    print("##################################")
    #print(len(examples))
    examples=tf.stack(examples)
    labels = tf.stack(labels)
    #examples=tf.expand_dims(examples, axis=0)
    #print(examples.shape)

    image_generator = image_datagen.flow(
                                        x=examples,
                                        batch_size = batch_size,
                                        shuffle=False,
                                        seed=seed,
                                        save_prefix=image_save_prefix,
                                        save_to_dir=save_to_dir)

    mask_generator = mask_datagen.flow(
                                        x=labels,
                                        batch_size=batch_size,
                                        shuffle=False,
                                        seed=seed,
                                        save_prefix=mask_save_prefix,
                                        save_to_dir=save_to_dir)

    while True:
        x = image_generator.next()

        y = mask_generator.next()
        #print("generator2 : " + str(x.shape), str(y.shape))
        yield (x, y)


#def gen_train_npy(image_path,mask_path,flag_multi_class = False,num_class = 2,image_prefix = "image",mask_prefix = "mask",image_as_gray = True,mask_as_gray = True):
#    image_name_arr = glob.glob(os.path.join(image_path,"%s*.png"%image_prefix))
#    image_arr = []
#    mask_arr = []
#    for index,item in enumerate(image_name_arr):
#        img = io.imread(item,as_gray = image_as_gray)
#        img = np.reshape(img,img.shape + (1,)) if image_as_gray else img
#        mask = io.imread(item.replace(image_path,mask_path).replace(image_prefix,mask_prefix),as_gray = mask_as_gray)
#        mask = np.reshape(mask,mask.shape + (1,)) if mask_as_gray else mask
        #img,mask = adjustData(img,mask,flag_multi_class,num_class)
#        image_arr.append(img)
#        mask_arr.append(mask)
#    image_arr = np.array(image_arr)
#    mask_arr = np.array(mask_arr)
#    return image_arr,mask_arr


#=====================#
#   Generate datas    #
#=====================#

'''train_dir       = os.path.join(os.getcwd(), '..', 'crossValidationv3','testDA')
input_folder    = 'input'
label_folder    = 'output'
save_dir        = 'visu_data_gen'
aug_dir         = os.path.join(train_dir, save_dir)


#transform param
data_gen_args = dict(brightness_range=[0.1,0.5],
                    rescale = 1.0 / 255,
                    rotation_range=20,
                    width_shift_range=0.2,
                    height_shift_range=0.2,
                    shear_range=0.2,
                    zoom_range=0.2,
                    horizontal_flip=True,
                    vertical_flip=True,
                    fill_mode='reflect',
                    preprocessing_function=get_random_eraser(p=0.5, s_l=0.02, s_h=0.4, r_1=0.3, r_2=1/0.3,
                  v_l=0, v_h=0, pixel_level=False))
#number of méta image you want to create.
number_batch = 10
__generator = trainset_generator(number_batch, train_dir, input_folder, label_folder,
                                      data_gen_args, save_to_dir=aug_dir)


for i,batch in enumerate(__generator):
    print(i)
    if(i >= number_batch):
        break'''
