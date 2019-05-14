# import seaborn as sns
import cv2
# import pandas as pd
# import numpy as np
# import tensorflow as tf
from tensorflow import keras
# from tensorflow.keras.layers import Conv2D, MaxPooling2D
# from tensorflow.keras.layers import Dense, Dropout, Flatten, Activation
# from tensorflow.keras.metrics import categorical_accuracy, top_k_categorical_accuracy, categorical_crossentropy
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
# from tensorflow.keras.optimizers import Adam
# from tensorflow.keras.applications import MobileNet
# from tensorflow.keras.applications.mobilenet import preprocess_input
import cv2
import numpy as np
# from IPython.display import display
# from PIL import Image
# from tensorflow.keras.applications import MobileNet
from keras.applications.inception_v3 import InceptionV3
from keras.applications.vgg16 import VGG16
# from keras.preprocessing import image
from keras.models import Model, load_model
from keras.layers import Dense, GlobalAveragePooling2D
import glob, os
# from keras import backend as K
# import time
# # start = time.datetime.now()


# base_model = InceptionV3(weights='imagenet', input_shape = (150,150, 3),include_top=False)
base_model = VGG16(weights='imagenet', input_shape = (150,150, 3),include_top=False)
x = base_model.output
x = GlobalAveragePooling2D()(x)
# let's add a fully-connected layer
x = Dense(1024, activation='relu')(x)
x=Dense(64, activation='relu')(x)

# and a logistic layer -- let's say we have 200 classes
predictions = Dense(units = 2, activation = 'softmax')(x)

# this is the model we will train
model = Model(inputs=base_model.input, outputs=predictions)

for layer in base_model.layers:
    layer.trainable = False
# model.compile(optimizer='adam', loss='binary_crossentropy',metrics = ['accuracy'])
model.compile(optimizer='adam', loss='categorical_crossentropy')
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

training_set = train_datagen.flow_from_directory('0510data/train',
                                                 target_size = (150,150),
                                                 batch_size = 32,
                                                 class_mode = 'categorical')

model.fit_generator(training_set,
                         steps_per_epoch = 1000,
                         epochs = 5)

model.save('0510dataVgg16.h5')

# model=load_model('0306data2.h5')
# for i, layer in enumerate(base_model.layers):
#    print(i, layer.name)

# for layer in model.layers[:249]:
#    layer.trainable = False
# for layer in model.layers[249:]:
#    layer.trainable = True

# from keras.optimizers import SGD
# model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy')
# model.fit_generator(training_set,
#                          steps_per_epoch = 1000,
#                          epochs = 10)
# model.save('0510datacompile.h5')



# model=load_model('0510data2adam1000.h5')

# files = glob.glob(os.path.join('0510data/test/1', "*"))
# # filename = '28-0012.jpg'
# c=0
# for filename in files:
#     # print(filename)
#     img = cv2.imread(filename)
#     img = cv2.resize(img,(150,150))
#     img = np.float32(img / 255)
#     img = np.reshape(img,[1,150,150,3])

#     pred = model.predict(img)
#     # print(pred)
#     if pred[0][0] < pred[0][1]:
#         print('1')
#         c=c+1
#     else:
#         print('0')
#     print(pred)
# print('%d in %d'  %(c, len(files)))