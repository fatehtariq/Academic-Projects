#!/usr/bin/env python
# coding: utf-8

# # **Artificial Neural Networks - Semester Project**

# ### **Handwritten Digits Classification with Deep Convolutional Neural Networks**

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')


# **Libraries**

# In[3]:


import numpy as np     
import matplotlib.pyplot as plt    
import random                     

from keras.datasets import mnist    
from keras.models import Sequential 

from keras.layers.core import Dense, Dropout, Activation 
from keras.utils import np_utils                

from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D, GlobalAveragePooling2D, Flatten
from keras.layers.normalization import BatchNormalization


# **Data Preprocessing**

# In[4]:


(X_train, y_train), (X_test, y_test) = mnist.load_data()


# Normalisation of Data

# In[5]:


X_train = X_train.reshape(60000, 28, 28, 1) 
X_test = X_test.reshape(10000, 28, 28, 1)

X_train = X_train.astype('float32')       
X_test = X_test.astype('float32')

X_train /= 255                             
X_test /= 255

print("Training matrix shape", X_train.shape)
print("Testing matrix shape", X_test.shape)


# In[6]:


nb_classes = 10 

Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)


# **Deep Convolutional Model**

# In[7]:


model = Sequential()

# Convolution Layer 1
model.add(Conv2D(32, (3, 3), input_shape=(28,28,1))) 
model.add(BatchNormalization(axis=-1))              
convLayer01 = Activation('relu')                    
model.add(convLayer01)

# Convolution Layer 2
model.add(Conv2D(32, (3, 3)))                   
model.add(BatchNormalization(axis=-1))              
model.add(Activation('relu'))
# Pooling Layer 1                 
convLayer02 = MaxPooling2D(pool_size=(2,2))          
model.add(convLayer02)

# Convolution Layer 3
model.add(Conv2D(64,(3, 3)))                      
model.add(BatchNormalization(axis=-1))              
convLayer03 = Activation('relu')               
model.add(convLayer03)

# Convolution Layer 4
model.add(Conv2D(64, (3, 3)))                       
model.add(BatchNormalization(axis=-1))            
model.add(Activation('relu'))
# Pooling Layer 2                    
convLayer04 = MaxPooling2D(pool_size=(2,2))        
model.add(convLayer04)
model.add(Flatten())                               

# Fully Connected Layer 5
model.add(Dense(512))                          
model.add(BatchNormalization())             
model.add(Activation('relu'))                

# Fully Connected Layer 6                       
model.add(Dropout(0.2))                           
model.add(Dense(10))                              
model.add(Activation('softmax'))                    


# In[8]:


model.summary()


# **Model Compilation**

# In[9]:


model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


# In[10]:


gen = ImageDataGenerator(rotation_range=8, width_shift_range=0.08, shear_range=0.3,
                         height_shift_range=0.08, zoom_range=0.08)

test_gen = ImageDataGenerator()


# In[11]:


train_generator = gen.flow(X_train, Y_train, batch_size=128)
test_generator = test_gen.flow(X_test, Y_test, batch_size=128)


# **Model Training**

# Batch Size: 128
# Steps per Epoch : 60,000/128 for 5 Epochs
# 

# Validation Steps: 10,000/128

# In[12]:


model.fit_generator(train_generator, steps_per_epoch=60000//128, epochs=5, verbose=1, 
                    validation_data=test_generator, validation_steps=10000//128)


# **Model Evaluation**

# In[13]:


score = model.evaluate(X_test, Y_test)
print('Test score:', score[0])
print('Test accuracy:', score[1])


# **MNIST Figure Representation**

# In[15]:


plt.figure()
plt.imshow(X_test[3].reshape(28,28), cmap='gray', interpolation='none')


# In[ ]:




