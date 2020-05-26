#!/usr/bin/env python
# coding: utf-8

# In[1]:


from keras.datasets import mnist


# In[2]:


df =mnist.load_data('mymnist.db')


# In[3]:


len(df)


# In[4]:


train, test=df


# In[5]:


len(train)


# In[6]:


X_train , y_train = train


# In[7]:


X_test , y_test = test


# In[8]:


X_train.shape


# In[9]:


X_test.shape


# In[10]:


train_X = X_train.reshape(-1, 28,28, 1)
test_X = X_test.reshape(-1, 28,28, 1)


# In[11]:


train_X = train_X.astype('float32')
test_X = test_X.astype('float32')


# In[13]:


from keras.utils import to_categorical


# In[14]:


import matplotlib.pyplot as plt 


# In[15]:


train_Y_one_hot = to_categorical(y_train)
test_Y_one_hot = to_categorical(y_test)


# In[16]:


import keras
from keras.datasets import fashion_mnist 
from keras.layers import Dense, Activation, Flatten, Conv2D, MaxPooling2D
from keras.models import Sequential
from keras.utils import to_categorical
import numpy as np


# In[17]:


model = Sequential()

model.add(Conv2D(64, (3,3), input_shape=(28, 28, 1)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64, (3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())
model.add(Dense(64))


# In[18]:


model.add(Dense(10))
model.add(Activation('softmax'))

model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(),metrics=['accuracy'])


# In[19]:


model.fit(train_X, train_Y_one_hot, batch_size=64, epochs=5)


# In[20]:


model.summary()


# In[21]:


test_loss, test_acc = model.evaluate(test_X, test_Y_one_hot)
print('Test loss', test_loss)
print('Test accuracy', test_acc)


# In[26]:


predictions = model.predict(test_X)
for i in range(50):
    print(np.argmax(np.round(predictions[i])))


# In[25]:


plt.imshow(test_X[1].reshape(28, 28), cmap = plt.cm.binary)
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[12]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




