import tensorflow as tf
import numpy
from matplotlib import pyplot
import dicom
import os
import cv2
import openpyxl




################ TensorFlow standard operations wrappers #####################
def weight(shape, name):
    initial = tf.truncated_normal(shape, stddev=0.01)
    w = tf.Variable(initial, name=name)
    return w

def bias(value, shape, name):
    initial = tf.constant(value, shape=shape)
    return tf.Variable(initial, name=name)

def conv2d(x, W, stride, padding):
    return tf.nn.conv2d(x, W, strides=[1, stride[0], stride[1], 1], padding=padding)

def max_pool2d(x, kernel, stride, padding):
    return tf.nn.max_pool(x, ksize=kernel, strides=stride, padding=padding)

def relu(x):
    return tf.nn.relu(x)


# In[56]:


################ Data creation functions #####################
def read_image(image_folder):
    
    """It reads all of image files in image_folder and process them
    
            Args:
                images_folder: path where all of image files locate
                
            Returns:
                img_set: the numpy array of the processed image set [num of image set, width, height, channels]
    """
   
    image_path_array = []  
    
    for dir_name, subdir_list, file_list in os.walk(image_folder):
        for filename in file_list:
            if ".dcm" in filename.lower():  # check whether the file's DICOM
                image_path_array.append(os.path.join(dir_name,filename))
   
    img_set = process_image(image_path_array)
    
    
    return img_set



def process_image(image_path_array):
    
    """It reads image set, it crops each of images to have dimension of 384*384px.
        After cropping, it shrinks each of images to have dimension of 192*192px.
        With them, it rotates and flip each of images for data augmentation.
        By processing images, image duplicated 12 times.
        
            Args:
                image_path_array: all of image files path
                
            Returns:
                img_set: the numpy array of the processed image set [num of image set, width, height, channels]
    """
    
    # Load dimensions based on the number of rows, columns, and duplicated images
    const_pixel_dims = (192, 192, len(image_path_array)*4*3)

    img_set = numpy.zeros(const_pixel_dims, dtype=numpy.float32)
    img_cnt = 0
    
    for filename in image_path_array:
        ds = dicom.read_file(filename)
        
        # 384*384 Crooping
        cropped_img = ds.pixel_array[50:434,50:434]
        # 192*192 Shrink
        resized_img = cv2.resize(cropped_img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)

        for j in range(4):
            rows,cols = resized_img.shape
            
            # Rotate (0,90,180,270)
            M = cv2.getRotationMatrix2D((cols/2,rows/2),90*j,1)
            img_rot = cv2.warpAffine(resized_img,M,(cols,rows))
            img_set[:, :, img_cnt] = img_rot
            img_cnt+=1
            
            # Flip left right
            img_flip1 = cv2.flip(img_rot,0)
            img_set[:, :, img_cnt] = img_flip1
            img_cnt+=1
            
            # Flip Up Down
            img_flip2 = cv2.flip(img_rot,1)
            img_set[:, :, img_cnt] = img_flip2
            img_cnt+=1
            
    img_set = numpy.reshape(img_set,[1,192,192,img_cnt])
    img_set = numpy.swapaxes(img_set,0,3)
    
    ### Image Print Test
    # img_sample = numpy.reshape(img_set[400],[192,192])
    # pyplot.imshow(img_sample,cmap='gray')
    # pyplot.show()
    
    return img_set



def read_label(label_file):
    
    """It reads a xlsx file in label_folder and process them (Data augmentation + One hot encoding)
    
            Args:
                label_file: path where a label file locates
                
            Returns:
                label_set: the numpy array of the processed label set [num of label set, result(2)]
    """
    
    csvfile = openpyxl.load_workbook(label_file)
    csvfile = csvfile.active
    label_set = []

    for cnt, row in enumerate(csvfile.rows,1):
        if(cnt!=1):
            # data augmentation (12 times)
            for i in range(12):
                label_set.append(row[1].value)

    #One Hot Encoding
    label_set = numpy.eye(2)[label_set]
    
    
    return label_set



def divide_trainset_testset(img_set,label_set):
    
    """It shufflse dataset and separate into train set(90%) and test set(10%)
    
            Args:
                img_set: the numpy array of the processed image set [num of image set, width, height, channels]
                label_set: the numpy array of the processed label set [num of label set, result(2)]
                
            Returns:
                train_img: the numpy array of the processed train image set [num of image set, width, height, channels]
                train_label: the numpy array of the processed train label set [num of label set, result(2)]
                test_img: the numpy array of the processed test image set [num of image set, width, height, channels]
                test_label: the numpy array of the processed test label set [num of label set, result(2)]
    """
    
    combined_set = list(zip(img_set,label_set))
    numpy.random.shuffle(combined_set)
    img_set[:],label_set[:] = zip(*combined_set)
    
    # Train Set: 90% / Test Set: 10%
    num_trainset = (int)(img_set.shape[0]*(0.9))
    train_img = img_set[0:num_trainset,:,:,:]
    test_img = img_set[num_trainset:,:,:]

    train_label = label_set[0:num_trainset,:]
    test_label = label_set[num_trainset:]
    
    return [train_img,train_label,test_img,test_label]


# In[1]:


# img = read_image("G:/NN Project/subset0/subset0")

# label = read_label("G:/NN Project/label.xlsx")

# a,b,c,d  = divide_trainset_testset(img,label)
