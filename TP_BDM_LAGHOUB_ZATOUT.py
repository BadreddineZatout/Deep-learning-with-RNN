#!/usr/bin/env python
# coding: utf-8

# In[323]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


# In[324]:


data = pd.read_csv(r"C:/Users/Laghoub/Desktop/epm_clean.csv", date_parser = True)
data


# In[325]:


data = data.drop(['student_Id'], axis = 1)


# In[326]:


data_training = data[0:25].copy()
data_training


# In[327]:


data_test = data[25:34].copy()
data_test


# In[328]:


scaler = MinMaxScaler()
data_training = scaler.fit_transform(data_training)
data_training


# In[329]:


X_train = []
y_train = []

for i in range(5, data_training.shape[0]):
    X_train.append(data_training[i-5:i])
    y_train.append(data_training[i,47])
    
X_train, y_train = np.array(X_train), np.array(y_train)


# In[330]:


X_train.shape, y_train.shape


# In[331]:


from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, Dropout


# In[332]:


model = Sequential()
model.add(SimpleRNN(units = 40, activation = 'relu',return_sequences = True,  input_shape = (X_train.shape[1], 48)))
model.add(Dropout(0.2))

model.add(SimpleRNN(units = 50, activation = 'relu', return_sequences = True))
model.add(Dropout(0.3))

model.add(SimpleRNN(units = 60, activation = 'relu'))
model.add(Dropout(0.4))


model.add(Dense(units =1))


# In[333]:


model.summary()


# In[334]:


model.compile(optimizer='adam', loss = 'mean_squared_error')
model.fit(X_train, y_train, epochs=40, batch_size=32)


# In[335]:


past_data = data.tail(5)
df = past_data.append(data_test, ignore_index = True)
df.head()


# In[336]:


inputsTest = scaler.transform(df)
inputsTest


# In[337]:


inputsTest.shape[0]


# In[338]:


X_test = []
y_test = []

for i in range(5, inputsTest.shape[0]):
    X_test.append(inputsTest[i-5:i])
    y_test.append(inputsTest[i, 47])

X_test, y_test = np.array(X_test), np.array(y_test)
X_test.shape, y_test.shape


# In[339]:


y_pred = model.predict(X_test)


# In[340]:


y_pred


# In[341]:


y_pred = y_pred
y_test = y_test
for i in range(0,8):
    if y_pred[i]>0.5:
        y_pred[i]=1
    else:
        y_pred[i]=0


# In[342]:


plt.figure(figsize=(14,5))
plt.plot(y_test, color = 'red', label = 'indicateur réel')
plt.plot(y_pred, color = 'blue', label = 'indicateur prédit')
plt.title('Prédiction de passage')
plt.xlabel('Student')
plt.ylabel('Passage')
plt.legend()
plt.show()


# In[ ]:





# In[ ]:




