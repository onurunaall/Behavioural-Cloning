!pip3 install imgaug #data augmentation and imaug augmentation >> keras augmentation

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import ntpath
import random
from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import Dense
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from imgaug import augmenters as iaa

datadir = 'track'
columns =['center', 'left', 'right', 'steering', 'throttle', 'reverse', 'speed']
data = pd.read_csv(os.path.join(datadir, 'driving_log.csv'), names=columns)
pd.set_option('display.max_colwidth', -1)
data.head()

def path_leaf(path):
  head, tail = ntpath.split(path)
  return tail

data['center'] = data['center'].apply(path_leaf)
data['left'] = data['left'].apply(path_leaf)
data['right'] = data['right'].apply(path_leaf)
data.head()

num_bin = 25
samples_per_bin = 400
hist, bins = np.histogram(data['steering'], num_bin)
center = (bins[:-1] + bins[1:]) * (0.5)
plt.bar(center, hist, width=0.05)
plt.plot((np.min(data['steering']), np.max(data['steering'])), (samples_per_bin, samples_per_bin))

print('total data: ', len(data))
remove_list = []
for j in range(num_bin):
  list_ = []
  for i in range(len(data['steering'])):
    if data['steering'][i] >= bins[j] and data['steering'][i] <= bins[j+1]:
      list_.append(i)
  list_ = shuffle(list_)
  list_ = list_[samples_per_bin:]
  remove_list.extend(list_)
  
print('removed: ', len(remove_list))
data.drop(data.index[remove_list], inplace=True)
print('remaining: ', len(data))

hist, _ = np.histogram(data['steering'], num_bin)
plt.bar(center, hist, width=0.05)
plt.plot((np.min(data['steering']), np.max(data['steering'])), (samples_per_bin, samples_per_bin))

print(data.iloc[1])
def load_img_steering(datadir, df):
  image_path = []
  steering = []
  for i in range(len(data)):
    indexed_data = data.iloc[i]
    center, left, right = indexed_data[0], indexed_data[1], indexed_data[2]
    # center image appending
    image_path.append(os.path.join(datadir, center.strip()))
    steering.append(float(indexed_data[3]))
    # left image appending
    image_path.append(os.path.join(datadir, center.strip()))
    steering.append(float(indexed_data[3]) + 0.15)
    # right image appending
    image_path.append(os.path.join(datadir, center.strip()))
    steering.append(float(indexed_data[3]) - 0.15)
  image_paths = np.asarray(image_path)
  steerings = np.asarray(steering)
  return image_paths, steerings

image_paths, steerings = load_img_steering(datadir + '/IMG', data) 
# image pats = the images(main data) // steerings = corresponding label

X_train, X_valid, y_train, y_valid = train_test_split(image_paths, steerings, test_size=0.2, random_state=6) #0.2 is reasonable split
print('Training Samples: {}\nValid Samples: {}'.format(len(X_train), len(X_valid)))

fig, axes = plt.subplots(1,2,figsize=(12, 4))
axes[0].hist(y_train, bins=num_bin, width=0.05, color='blue')
axes[0].set_title('Training Set')
axes[1].hist(y_valid, bins=num_bin, width=0.05, color='red')
axes[1].set_title('Validation Set')

def zoom(image):
  zoom = iaa.Affine(scale=(1, 1.3)) #zooming up to 30% of image
  image = zoom.augment_image(image)
  return image

image = image_paths[random.randint(0,1000)]
original_image = mpimg.imread(image)
zoomed_image = zoom(original_image)

fig, axs = plt.subplots(1,2,figsize=(15,10))
fig.tight_layout()

axs[0].imshow(original_image)
axs[0].set_title('Original Image')
axs[1].imshow(zoomed_image)
axs[1].set_title('Zoomed Image')

def pan(image):
  pan = iaa.Affine(translate_percent={"x": (-0.1,0.1), "y": (0.1,0.1)})
  image = pan.augment_image(image)
  return image

image = image_paths[random.randint(0,1000)]
original_image = mpimg.imread(image)
panned_image = pan(original_image)

fig, axs = plt.subplots(1,2,figsize=(15,10))
fig.tight_layout()

axs[0].imshow(original_image)
axs[0].set_title('Original Image')
axs[1].imshow(panned_image)
axs[1].set_title('Panned Image')

def img_random_brightness(image):
  brightness = iaa.Multiply((0.2, 1.2)) #darker images are more effective in comparison to more bright ones
  image = brightness.augment_image(image)
  return image

image = image_paths[random.randint(0, 1000)]
original_image = mpimg.imread(image)
brightness_altered_image = img_random_brightness(original_image)

fig, axs = plt.subplots(1,2,figsize=(15,10))
fig.tight_layout()

axs[0].imshow(original_image)
axs[0].set_title('Original Image')
axs[1].imshow(panned_image)
axs[1].set_title('Brightness Altered Image')

def img_random_flip(image, steering_angle):
  image = cv2.flip(image, 1) # 1 means horizontal flip
  steering_angle = -steering_angle
  return image, steering_angle

random_index = random.randint(0, 1000)
image = image_paths[random_index]
steering_angle = steerings[random_index]

original_image = mpimg.imread(image)
flipped_image, flipped_steering_angle = img_random_flip(original_image,steering_angle)

fig, axs = plt.subplots(1,2,figsize=(15,10))
fig.tight_layout()

axs[0].imshow(original_image)
axs[0].set_title('Original Image - ' + 'Steering Angle:' + str(steering_angle))
axs[1].imshow(flipped_image) 
axs[1].set_title('Flipped Image - '+ 'Flipped Steering Angle: '+ str(flipped_steering_angle))

def random_augment(image, steering_angle):
  image = mpimg.imread(image)
  if np.random.rand() < 0.5:
    #each augmentation will be applied only 50%
    image = pan(image)
  if np.random.rand() < 0.5:
    #each augmentation will be applied only 50%
    image = zoom(image)
  if np.random.rand() < 0.5:
    #each augmentation will be applied only 50%
    image = img_random_brightness(image)
  if np.random.rand() < 0.5:
    #each augmentation will be applied only 50%
    image, steering_angle = img_random_flip(image, steering_angle)
  return image, steering_angle

number_of_columns = 2
number_of_rows = 10
fig, axs = plt.subplots(number_of_rows, number_of_columns, figsize=(15,50))
fig.tight_layout()

for i in range(10):
  randnum = random.randint(0, len(image_paths)-1)
  random_image = image_paths[randnum]
  random_steering = steerings[randnum]

  original_image = mpimg.imread(random_image)
  augmented_image, steering = random_augment(random_image, random_steering)

  axs[i][0].imshow(original_image)
  axs[i][0].set_title('Original Image')
  axs[i][1].imshow(augmented_image)
  axs[i][1].set_title ('Augmented Image')

def img_preprocess(img):
  # cropping unnecessary part of image
  img = img[60:135,:,:] # height,width,channel
  img = cv2.cvtColor(img,cv2.COLOR_RGB2YUV) # yuv format due to nvidia arch.
  img = cv2.GaussianBlur(img,(3,3),0) #reducing noise w/GaussianBlur
  img = cv2.resize(img,(200,66)) #for nvidia model arch.
  img = img/255 #normalizing the variables
  return img

image = image_paths[100] # choosing a random image to check the effects of preprocessing
original_image = mpimg.imread(image)
preprocessed_image = img_preprocess(original_image)

fig, axs = plt.subplots(1,2,figsize=(15, 10))
fig.tight_layout()

axs[0].imshow(original_image)
axs[0].set_title('Original Image')
axs[1].imshow(preprocessed_image)
axs[1].set_title('Preprocessed Image')

def batch_generator(image_paths, steering_ang, batch_size, istraining):
  #validation data should not be augmented
  #istraining will be 1 when training data is fed and 0 when valid data is fed
  while True:
    batch_img = []
    batch_steering = []
    
    for i in range(batch_size):
      random_index = random.randint(0, len(image_paths)-1)
      if istraining:
        im, steering = random_augment(image_paths[random_index], steering_ang[random_index])
      else:
        im = mpimg.imread(image_paths[random_index])
        steering = steering_ang[random_index]

      im = img_preprocess(im)
      batch_img.append(im)
      batch_steering.append(steering)
    yield (np.asarray(batch_img), np.asarray(batch_steering))

X_train_gen, y_train_gen = next(batch_generator(X_train, y_train, 1, 1))
X_valid_gen, y_valid_gen = next(batch_generator(X_valid, y_valid, 1, 0))

fig, axs = plt.subplots(1, 2, figsize=(15, 10))
fig.tight_layout()

axs[0].imshow(X_train_gen[0])
axs[0].set_title('Training Image')
axs[1].imshow(X_valid_gen[0])
axs[1].set_title('Validation Image')

# X_train = np.array(list(map(img_preprocess,X_train))) #all the X_train variables will be preprocessed w/ that way
# X_valid = np.array(list(map(img_preprocess,X_valid)))
# since all the data generation process is done in the generator,
# there is no need to use these two commands

# plt.imshow(X_train[random.randint(0,len(X_train)-1)]) # selecting and plotting a random image from the X_train set
# plt.axis('off')
# print(X_train.shape) #(# of images, y length, x length, # of channel)

def nvidia_model():
  # droupout layers has been removed because of over-fitting
  model = Sequential()
  model.add(Conv2D(24, kernel_size=(5,5), strides=(2,2), input_shape=(66, 200, 3), activation='elu'))
  model.add(Conv2D(36, kernel_size=(5,5), strides=(2,2), activation='elu'))
  model.add(Conv2D(48, kernel_size=(5,5), strides=(2,2), activation='elu'))
  model.add(Conv2D(64, kernel_size=(3, 3), activation='elu'))
  model.add(Conv2D(64, kernel_size=(3, 3), activation='elu'))
  # model.add(Dropout(0.5))
  
  model.add(Flatten())
  model.add(Dense(100, activation='elu'))
  # model.add(Dropout(0.5))

  model.add(Dense(50, activation='elu'))
  # model.add(Dropout(0.5))

  model.add(Dense(10, activation ='elu'))
  # model.add(Dropout(0.5))
  
  model.add(Dense(1))

  optimizer = Adam(lr=1e-3)
  model.compile(loss='mse', optimizer=optimizer)
  return model

model = nvidia_model()
print(model.summary())

history = model.fit_generator(batch_generator(X_train, y_train, 64, 1),
                                  steps_per_epoch=500, 
                                  epochs=10,
                                  validation_data=batch_generator(X_valid, y_valid, 64, 0),
                                  validation_steps=200,
                                  verbose=1,
                                  shuffle=1)
#batch_size=100 means batch_generator will create new 100 images per step
#steps_per_epoch=300 means there will be 300 steps before completing an epoch
#which makes our system will have 30000 images per epoch

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['training', 'validation'])
plt.title('loss')
plt.xlabel('# of epochs')

model.save("model.h5")

from google.colab import files
files.download('model.h5')
