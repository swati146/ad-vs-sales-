#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[4]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[8]:


df=pd.read_csv(r"C:\Users\91805\OneDrive\Desktop\Book1.csv", encoding='latin-1')


# In[9]:


print(df.shape) 


# In[10]:


print (df)


# In[11]:


print (df.head())


# In[12]:


df.columns = ['Sales', 'Advertising']


# In[13]:


print(df.head())


# In[14]:


df.info()


# In[15]:


df.describe()


# In[16]:


x=df['Sales'].values
y=df['Advertising'].values


# In[17]:


plt.scatter(x, y, color = 'blue', label='Scatter Plot')
plt.suptitle('Relationship between Sales and Advertising')
plt.xlabel('Sales')
plt.ylabel('Advertising')
plt.legend(loc=4)
plt.show()


# In[18]:


print(x.shape)
print(y.shape)


# In[19]:


X=x.reshape(-1,1)
Y=y.reshape(-1,1)


# In[20]:


print(X.shape)
print(Y.shape)


# In[21]:


from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X, Y, test_size=0.33, random_state=42)


# In[22]:


print(X_train.shape)
print(Y_train.shape)
print(X_test.shape)
print(Y_test.shape)


# In[23]:


from sklearn.linear_model import LinearRegression
lm=LinearRegression()
lm.fit(X_train,Y_train)
Y_pred=lm.predict(X_test)


# In[24]:


a=lm.coef_,
b=lm.intercept_,
print("Estimated model slope a:", a)
print("Estimated model slope b:", b)


# In[25]:


lm.predict(X)[0:5]


# In[26]:


from sklearn.metrics import mean_squared_error
mse=mean_squared_error(Y_test, Y_pred)
rmse=np.sqrt(mse)
print(rmse)


# In[27]:


from sklearn.metrics import r2_score
print ("R2 Score value: {:.4f}".format(r2_score(Y_test, Y_pred)))


# In[28]:


plt.scatter(X, Y, color = 'blue', label='Scatter Plot')
plt.plot(X_test, Y_pred, color = 'black', linewidth=3, label = 'Regression Line')
plt.suptitle('Relationship between Sales and Advertising')
plt.xlabel('Sales')
plt.ylabel('Advertising')
plt.legend(loc=4)
plt.show()


# In[29]:


plt.scatter(lm.predict(X_train), lm.predict(X_train) - Y_train, color = 'red', label = 'Train data')
plt.scatter(lm.predict(X_test), lm.predict(X_test) - Y_test, color = 'blue', label = 'Test data')
plt.hlines(xmin = 0, xmax = 50, y = 0, linewidth = 3)
plt.suptitle('Residual errors')
plt.legend(loc = 4)
plt.show()


# In[32]:


print("Training set score: {:.4f}".format(lm.score(X_train,Y_train)))

print("Test set score: {:.4f}".format(lm.score(X_test,Y_test)))


# In[ ]:





# In[ ]:




