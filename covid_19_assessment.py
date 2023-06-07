"""
Assessment: Covid-19 Prediction with RNN (LSTM)
"""
#%%
#1. Import packages
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
import os, datetime
from tensorflow.keras import callbacks
import tensorboard



PATH = os.getcwd()
CSV_PATH_TRAIN = os.path.join(PATH,"cases_malaysia_train.csv")
CSV_PATH_TEST = os.path.join(PATH,"cases_malaysia_test.csv")
df_train = pd.read_csv(CSV_PATH_TRAIN)
df_test = pd.read_csv(CSV_PATH_TEST)
# %%
#2. Data Inspection
import matplotlib.pyplot as plt

df_train.head()
#%%
df_train.tail()
#%%
df_train.info()
#%%
df_train.isna().sum()
#%%
df_test.isna().sum()
#%%
# Calculate the mean of each column excluding the date column
test_column_means = df_test.drop('date', axis=1).mean()


# Replace NaN values in the testing dataset with the column means
df_test = df_test.fillna(test_column_means)


#%%
df_test.isna().sum()
# %%
df_test.head(65)
#%%
#check for non-numeric 
non_numeric_values = df_train["cases_new"].loc[df_train["cases_new"].apply(lambda x: not x.isnumeric())].unique()
print(non_numeric_values)

#%%
#remove non-numeric row and change column as float type.
df_train = df_train.loc[df_train["cases_new"].apply(lambda x: x.isnumeric())]
df_train["cases_new"] = df_train["cases_new"].astype(float)


#%%
# Plot a graph to see the "cases_new" 
df_disp=df_train[:200]
plt.figure()
plt.plot(df_disp['cases_new'])
plt.show()
# %%
#3. Feature selection
from sklearn.preprocessing import MinMaxScaler

X= df_train["cases_new"]
mms=MinMaxScaler()
X =mms.fit_transform(np.expand_dims(X,axis=-1))


# %%
# Data windowing
window_size= 30
X_train=[]
y_train=[]

for i in range(window_size, len(X)):
    X_train.append(X[i-window_size:i])
    y_train.append(X[i])

X_train= np.array(X_train)
y_train=np.array(y_train)
# %%
# concatenate train and test data target together
df_cat= pd.concat((df_train["cases_new"], df_test["cases_new"]))
#Method 1
length_days= len(df_cat)- len(df_test)- window_size
tot_input= df_cat[length_days:]
# Method 2
length_days= window_size + len(df_test)
tot_input= df_cat[-length_days:]

data_test = mms.transform(np.expand_dims(tot_input, axis=-1))

X_test=[]
y_test=[]
for i in range(window_size, len(data_test)):
    X_test.append(data_test[i-window_size:i])
    y_test.append(data_test[i])

X_test=np.array(X_test)
y_test=np.array(y_test)
# %%
# 4. Model development
from tensorflow.keras import Sequential, Input
from tensorflow.keras.layers import LSTM,Dropout,Dense
from tensorflow.keras.utils import plot_model

input_shape = np.shape(X_train)[1:]

model = Sequential()
model.add(Input(shape=(input_shape)))
model.add(LSTM(64,return_sequences=True))
model.add(Dropout(0.3))
model.add(LSTM(64, return_sequences=True))
model.add(Dropout(0.3))
model.add(LSTM(64))
model.add(Dropout(0.3))
model.add(Dense(1, activation='relu'))
model.summary()
# %%
#Compile the model
model.compile(optimizer='adam', loss='mse', metrics=['mae'])
# %%
# Create a TensorBoard callback object for the usage of TensorBoard
base_log_path = r"tensorboard_logs\covid19"
if not os.path.exists(base_log_path):
    os.makedirs(base_log_path)
log_path = os.path.join(base_log_path,datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
tb = callbacks.TensorBoard(log_path)
#%%
# 5. Model training
history= model.fit(X_train, y_train, epochs=20, callbacks=[tb])
# %%
#6. Model evaluation
print(history.history.keys())



# %%
#Deploy the model
y_pred=model.predict(X_test)
# %%
# Perform inverse transform
actual_cases= mms.inverse_transform(y_test)
predicted_cases= mms.inverse_transform(y_pred)

# %%
#plot actual vs predicted
plt.figure()
plt.plot(actual_cases,color='red')
plt.plot(predicted_cases,color='blue')
plt.xlabel("Days")
plt.ylabel("Number of cases")
plt.legend(['Actual','Predicted'])
# %%
mask = actual_cases != 0  # Create a mask to exclude zero values
absolute_errors = np.abs((actual_cases - predicted_cases) / actual_cases)
absolute_errors = absolute_errors[mask]  # Apply the mask

# Calculate MAPE
mape = np.mean(absolute_errors) * 100

# Print MAPE
print("MAPE:", mape)
# %%
