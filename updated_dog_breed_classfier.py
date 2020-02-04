import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.utils import shuffle
import cv2
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D, GlobalAveragePooling2D,Activation,BatchNormalization
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from skimage.transform import rotate, AffineTransform, warp
from skimage.util import random_noise
import urllib.request
import random
import requests
import tarfile


#Downloader

def downloader(image_url):
    file_name = random.randrange(1,10000)
    full_file_name = "/content/"+str(file_name) + '.jpg'
    urllib.request.urlretrieve(image_url,full_file_name)
    return full_file_name

# Downloading and Extracting the dataset

url = "http://vision.stanford.edu/aditya86/ImageNetDogs/images.tar"
PATH_dir = '/content/dataset'

r = requests.get(url, allow_redirects=True)

open('image_dataset.tar', 'wb').write(r.content)


my_tar = tarfile.open('image_dataset.tar')
my_tar.extractall(PATH_dir)
my_tar.close()
print("Download of dataset complete")


#Augmnetation
def augment(img,total_bool = True):

    #rotation
    total_image =[]
    rotated_images=[]
    for i in [45,135]:
        total_image.append(rotate(img, angle=i))
        rotated_images.append(rotate(img, angle=i))


    #flips
    flip_images = []
    flip_images.append(np.fliplr(img))

    total_image.append(np.fliplr(img))


    #noise
    noisy_image=[]
    noisy_image.append(random_noise(img))
    total_image.append(random_noise(img))

    #blurry
    blur_image=[]
    blur_image.append(cv2.GaussianBlur(img, (11,11),0))
    total_image.append(cv2.GaussianBlur(img, (11,11),0))

    if (total_bool==False):
        return rotated_images,flip_images,noisy_image,blur_image
    else:
        return total_image



    n_breeds = 7
    image_in_each_breed = 100


n_breeds = 7
def extract_data(start,batch_size):
    PATH_IMG = '/content/dataset/Images/'
    imgs_data = []
    img_label_index = []
    n_breeds = 7
    image_in_each_breed = 100
# Reading the extracted images
    k=0
    for i in os.listdir(PATH_IMG)[1:n_breeds + 1]:
        p=0
        k=k+1
        for j in os.listdir(PATH_IMG+i)[start:start+batch_size]:
            imgs_data.append(list(cv2.resize(cv2.imread(PATH_IMG+i+'/'+j)/255,(256,256),interpolation = cv2.INTER_AREA)))
            p = p+1
        img_label_index.append(p)
    modified_img_data = np.asarray(imgs_data)


# APPLYING AUGMENTATION AND LABELING THE DATA

    modified_imgs_data_ = []
    tot_img = modified_img_data.shape[0]
    for i in modified_img_data:
        for j in augment(i):
            modified_imgs_data_.append(list(j))

    modified_img_data = np.asarray(modified_imgs_data_)

    train_label_ = []
    # NUMBER OF IMAGES PRODUCED BY AUGMENTATION (here 5)
    augumentation_factor = 5
    p = 0
    for i in img_label_index:
        for j in range(i*augumentation_factor):
            train_label_.append(p)
        p = p+1
    train_label = np.asarray(train_label_)
    print(modified_img_data.shape)
    return  modified_img_data,train_label


# CREATING THE MODEL

model = Sequential()

model.add(Conv2D(16, (3, 3), padding='same', use_bias=False, input_shape=(256, 256, 3)))
model.add(BatchNormalization(axis=3, scale=False))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(4, 4), strides=(4, 4), padding='same'))
model.add(Dropout(0.2))

model.add(Conv2D(32, (3, 3), padding='same', use_bias=False))
model.add(BatchNormalization(axis=3, scale=False))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(4, 4), strides=(4, 4), padding='same'))
model.add(Dropout(0.2))

model.add(Conv2D(64, (3, 3), padding='same', use_bias=False))
model.add(BatchNormalization(axis=3, scale=False))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(4, 4), strides=(4, 4), padding='same'))
model.add(Dropout(0.2))

model.add(Conv2D(128, (3, 3), padding='same', use_bias=False))
model.add(BatchNormalization(axis=3, scale=False))
model.add(Activation("relu"))
model.add(Flatten())
model.add(Dropout(0.2))

model.add(Dense(512, activation='relu'))
model.add(Dense(n_breeds, activation='softmax'))

# COMPILING THE model

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

pos = 0
for epochs in range(10):
    print(str(epochs) + " of 10 MAIN EPOCHS")
    out1 = np.zeros((1,256,256,3))
    out2 = np.zeros((1))
    out1,out2 = extract_data(pos,10)

    pos = pos+10

    # SHUFFLING DATASET

    modified_img_data , train_label = shuffle(out1,out2)

    # FITTING DATA INTO THE model

    history = model.fit(modified_img_data, train_label , epochs=10 , validation_split=0.3 , shuffle=True,)

# PLOTTING THE ACCURACY AND LOSS OF THE model

plt.plot(history.history['accuracy'])
plt.plot(history.history['loss'])

# SAVING THE MODEL

model.save('dog_classifier_model.h5')
