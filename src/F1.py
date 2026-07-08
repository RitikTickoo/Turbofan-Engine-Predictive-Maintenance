###  importing all the important libraries and modules for the project ###

# for handling numerical operations
import numpy as np
from numpy import isin
import pandas as pd

import os
import random
import tensorflow as tf

# for visualising the patterns in the dataset
import seaborn as sns
import matplotlib.pyplot as plt

# for the purpose of building machine learning models, splitting the dataset and evaluating results 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error
from sklearn.ensemble import RandomForestRegressor

# for model building through deep learning
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout,Input
from tensorflow.keras.layers import Bidirectional
from tensorflow.keras.callbacks import EarlyStopping


SEED = 2002

os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# naming of the columns for easy reference
columns=['unit_number','cycles','operational_setting_1','operational_setting_2','operational_setting_3']+ \
     [f'sensor_measurement{i}' for i in range(1,22)]

# LOADING OF THE DATASET FOR THE INITIAL INSPECTION
data=pd.read_csv('train_FD001.txt',sep=r'\s+',names=columns,header=None)

# DISPLAYING THE DATASET JUST TO UNDERSTAND THE SHAPE, SIZE & QUALITY OF THE DATASET
print(data.head())
print(data.info())
print(data.shape)
print('DUPLICATED VALUES ARE AS FOLLOWS:',data.duplicated().sum())
print( data['unit_number'].value_counts())
max_cycles=(data.groupby('unit_number')['cycles'].max().reset_index())

# DISPLAY THE DESCRIPTIVE STATISTICS OF THE DATASET SUCH AS CENTRAL TENDENCY & DISPERSION
print(max_cycles.describe())

print('---------')

### DATA VISUALIZATION ###

# GENERATING HISTOGRAM TO UNDERSTAND THE DISTRIBUTION OF LIFE CYCLES OF ENGINES
plt.figure(figsize=(10,6))
plt.hist(max_cycles['cycles'],bins=10,color='teal')
plt.title('Distribution of Engine life cycles')
plt.xlabel('max cycles')
plt.ylabel('frequency')
plt.show()

# SELECTING ENGINE 1 FOR EXPLORING SENSOR TREND ANALYSIS
engine_1=data[data['unit_number']==1]

# SENSOR 1-8
plt.figure(figsize=(10,6))
for i in range(1,9):
     plt.plot(engine_1['cycles'],engine_1[f'sensor_measurement{i}'],label=f'sensor{i}')
plt.title('Engine_1 vs sensor 1_8 trend analysis')
plt.xlabel('cycles')
plt.ylabel('sensors measurement')
plt.legend(bbox_to_anchor=(1.10,1),loc='upper right')
plt.show()

# SENSOR 9-16
plt.figure(figsize=(10,6))
for i in range(9,17):
     plt.plot(engine_1['cycles'],engine_1[f'sensor_measurement{i}'],label=f'sensor{i}')
plt.title('Engine_1 vs sensor 9_16 trend analysis')
plt.xlabel('cycles')
plt.ylabel('sensors measurement')
plt.legend(bbox_to_anchor=(1.10,1),loc='upper right')
plt.show()

# SENSOR 17-21
plt.figure(figsize=(10,6))
for i in range(17,22):
     plt.plot(engine_1['cycles'],engine_1[f'sensor_measurement{i}'],label=f'sensor{i}')
plt.title('Engine_1 vs sensor 7_21 trend analysis')
plt.xlabel('cycles')
plt.ylabel('sensors measurement')
plt.legend(bbox_to_anchor=(1.10,1),loc='upper right')
plt.show()

# SENSOR TREND ANALYSIS OF ENGINE_1 WITH SINGLE SENSOR
plt.figure(figsize=(10,6))
plt.plot(engine_1['cycles'], engine_1['sensor_measurement7'], label='Sensor 7')
plt.title('Engine 1 vs Sensor 7 Trend Analysis')
plt.xlabel('Cycles')
plt.ylabel('Sensor Measurement')
plt.legend(bbox_to_anchor=(1.10,1), loc='upper right')
plt.show()

# CALCUATION OF REMANING USEFUL LIFE OF EACH ENGINE
max_cycles.columns=['unit_number','max_cycles']
data=data.merge(max_cycles, on='unit_number')
data['RUL']=data['max_cycles']-data['cycles']
# CAP HIGH RUL VALUES AT 125
RUL_CAP = 125
data['RUL'] = data['RUL'].clip(upper=RUL_CAP)
print(data[["unit_number",'max_cycles','RUL']].head(10))
print('---------')

# Check correlation of all numeric features with RUL
# TO VERIFY WHICH ATTRIBUTES ARE STRONGLY/WEAKLY RELATED WITH OUR TARGET VARIABLE(RUL)
correlation=round(data.corr()['RUL'].sort_values(),2)
print( 'correlation values',correlation)
print('---------')


# Visualise correlation of selected sensor features with RUL using a heatmap
sensor_columns = [f'sensor_measurement{i}' for i in [2,3,4,7,8,11,12,13,15,17,20,21]]
correlation_with_rul = data[sensor_columns + ['RUL']].corr()[['RUL']]
plt.figure(figsize=(8, 10))
sns.heatmap(correlation_with_rul, annot=True, fmt='.2f', cmap='coolwarm')
plt.title('Sensor Correlation with RUL')
plt.show()


### Preprocessing ###

# COPY OF ORIGINAL DATASET CREATED 
data_copy=data.copy()

# REMOVING  UNNECESSARY ATTRIBUTES BEFORE MODEL BUILDING
drop_sensors=[f'sensor_measurement{i}' for i in [1,5,10,16,18,19,9,14,6]]
data_copy=data_copy.drop(columns=drop_sensors)
print([col for col in data_copy.columns if 'sensor' in col])
print(data_copy.columns)

# REMOVING ATTRIBUTE "max_cycles" TO PREVENT DATA LEAKAGE
data_copy=data_copy.drop(columns='max_cycles')
print(data_copy.columns)

# SELECTION OF INDEPENDENT FEATURES AND DEPENDENT TARGET VARIABLE FOR MODEL TRAINING
feature_columns=[col for col in data_copy if col not in ['unit_number','RUL']]
target_column='RUL'
print('Feature Columns')
print(feature_columns)
print('number of features:',len(feature_columns))
print('target column:', target_column)

# SPLITTING THE DATASET INTO TEST-TRAIN SETS ON THE BASIS OF ENGINE UNIT NUMBER
# THIS AVOIDS DATA LEAKAGE BY PREVENTING THE ENGINE WITH SAME NUMBER TO APPEAR IN TRAINING & TESTING SPLIT
units=data_copy['unit_number'].unique()
train_units,test_units=train_test_split(units,test_size=0.2,random_state= 2002)
train_data=data_copy[data_copy['unit_number'].isin(train_units)].copy()
test_data=data_copy[data_copy['unit_number'].isin(test_units)].copy()

print('Training engines :',len(train_units))
print('Testing engines :',len(test_units))
print('train data shape :',train_data.shape)
print('test data shape :',test_data.shape)

#----- scaling------

# SCALING BOTH TRAINING AND TESTING SET USING MINMAX SCALER
scaler=MinMaxScaler()
train_data[feature_columns]=scaler.fit_transform(train_data[feature_columns])
test_data[feature_columns]=scaler.transform(test_data[feature_columns])

x_train=train_data[feature_columns]
y_train=train_data[target_column]

x_test=test_data[feature_columns]
y_test=test_data[target_column]

print('x train :',x_train.shape)
print('y train :',y_train.shape)
print('x test :',x_test.shape)
print('y test :',y_test.shape)



# LINEAR REGRESSION MODEL WAS USED AS BASELINE MODEL
# TO IDENTIFY THE LINEAR RELATIONSHIP BETWEEN INDEPENDENT FEATURES AND DEPENDENT TARGET VARIABLE(RUL)
model_1=LinearRegression()
# TRAIN LINEAR REGRESSION MODEL USING TRAINING FEATURES AND TARGET VALUES
model_1.fit(x_train,y_train)
# PREDICT RUL VALUES FOR TEST DATASET
y_pred_model_1=model_1.predict(x_test)
# EVALUATING THE MODEL THROUGH MAE,RMSE & R2 SCORE
mae=mean_absolute_error(y_test,y_pred_model_1)
rmse=np.sqrt(mean_squared_error(y_test,y_pred_model_1))
r2=r2_score(y_test,y_pred_model_1)

print('mean absolute error for linear regression:',mae)
print("root mean squared error for linear regression :",rmse)
print("r2 :",r2)


# RANDOM FOREST FOR PERFORMANCE IMPROVEMENT AND PREVENT DATA LEAKAGE 
model_2=RandomForestRegressor(
     n_estimators=100,
     random_state=2002,
     n_jobs=-1
)
# TRAINING RANDOM FOREST MODEL USING TRAINING FEATURES AND TARGET VALUES
model_2.fit(x_train,y_train)

# PREDICT RUL VALUES FOR TEST DATASET
y_pred_model_2=model_2.predict(x_test)

# EVALUATING THE MODEL THROUGH MAE,RMSE & R2 SCORE
mae_model_2=mean_absolute_error(y_test,y_pred_model_2)
rmse_model_2=np.sqrt(mean_squared_error(y_test,y_pred_model_2))
r2_model_2=r2_score(y_test,y_pred_model_2)

print('mean absolute error for random forest:',mae_model_2)
print('root mean squared error for random forest:',rmse_model_2)
print('r2 for rf:',r2_model_2)  

#  LSTM MODELS ###

# DEFINE HOW MANY NUMBER OF CYCLES LSTM SHOULD LOOK AT 
#TO PREDICT THE CURRENT RUL CYCLE
sequence_length = 30

# FUNCTIN THAT CONVERTS 2D DATA INTO 3D FORMAT 
# LSTM UNDERSTANDS ONLY 3D DATA(SAMPLES,TIME-STEPS,FEATURES)
def create_sequence(data_copy, feature_columns, target_column, sequence_length):
     x=[]
     y=[]

# SORT THE CYCLES OF EACH ENGINE SEPARATELY 
# ALLOWS LSTM TO LEARN THE PATTERN OF ENGINE DEGRADATION
     for i in data_copy['unit_number'].unique():
          engine_data=data_copy[data_copy['unit_number']==i].sort_values('cycles')

# SEPARATE INDEPENDENT FEATURES AND DEPENDENT TARGET VARIABLE(RUL)
          features=engine_data[feature_columns].values
          Target=engine_data[target_column].values

# creating rolling windows of previous 30 cycles as input sequence
          for i in range(sequence_length,len(engine_data)):
               x.append(features[i-sequence_length:i])
               y.append(Target[i])
          
# RETURN THE INPUT SEQUENCE AS 3D ARRAY FOR LSTM 
# AND TARGET VALUES AS 1D NUMPY ARRAY
     return np.array(x),np.array(y)

# lstm train and test sequences 
x_train_lstm,y_train_lstm=create_sequence(train_data, feature_columns, target_column, sequence_length)

x_test_lstm,y_test_lstm=create_sequence(test_data, feature_columns, target_column, sequence_length)


print('shape of x_train_lstm',x_train_lstm.shape)
print('shape of y_train_lstm',y_train_lstm.shape)
print('shape of x_test_lstm',x_test_lstm.shape)
print('shape of y_test_lstm',y_test_lstm.shape)


### LSTM MODEL BUILDING ###

model_3=Sequential([
# Defines the input shape for the LSTM model.
     Input(shape=(sequence_length,len(feature_columns))),
# LEARNS ENGINE PATTERNS FROM SEQUENCE DATA 
     LSTM(64),
# Dropout layer to reduce overfitting by randomly ignoring
# 20% of neurons during training.
     Dropout(0.2),
# CREATES A LAYER WITH 32 NEURONS TO LEARN THE PATTERNS
# AND USE "RELU" TO CONVERT NEGATIVE VALUES TO ZERO
     Dense(32,activation='relu'),
# CREATES A FINAL OUTPUT LAYER WITH ONE NEURON
# IT COMBINES THE 32 VALUES FROM THE PREVIOUS LAYER USING LEARNED WEIGHTS AND BIAS
# AND OUTPUTS ONE PREDICTED RUL VALUE
     Dense(1)
])

 # training model#

model_3.compile(
# ADAM OPTIMIZER UPDATES THE MODEL WEIGHTS DURING TRAINING
# TO REDUCE THE PREDICTION ERROR.
     optimizer='adam',
# MODEL WILL USE "MSE" TO IMPROVE ITS PERFORMANCE
     loss='mse',
# MAE SHOWS THE DIFFERNCE BETWEEN PREDICTED VALUE AND ACTUAL VALUES
     metrics=['mae']
)

early_stopping_lstm = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)
history = model_3.fit(
# TRAINING INPUT SEQUENCES.
     x_train_lstm,

# ACTUAL RUL VALUES FOR TRAINING THE MODEL
     y_train_lstm,

# THE MODEL CAN GO THROUGH THE TRAINING DATA UP TO 50 TIMES.
# EARLY STOPPING MAY STOP TRAINING BEFORE 50 EPOCHS
# IF THE VALIDATION LOSS DOES NOT IMPROVE.
     epochs=50,

# THE MODEL UPDATES WEIGHTS AFTER PROCESSING 32 SAMPLES AT A TIME.
     batch_size=32,

# 20% OF THE TRAINING DATA IS USED FOR VALIDATION DURING TRAINING.
     validation_split=0.2,

# SHUFFLE IS FALSE BECAUSE LSTM USES SEQUENCE/TIME-BASED DATA.
# KEEPING THE ORDER HELPS PRESERVE THE SEQUENCE STRUCTURE.
     shuffle=False,

# STOPS TRAINING IF VALIDATION LOSS DOES NOT IMPROVE
# AND RESTORES THE BEST MODEL WEIGHTS.
     callbacks=[early_stopping_lstm]
)

# TESTING THE MODEL ON UNSEEN DATA
y_pred_model_3 = model_3.predict(x_test_lstm).flatten()

# EVALUATION OF THE MODEL
mae_model_3 = mean_absolute_error(y_test_lstm, y_pred_model_3)
rmse_model_3 = np.sqrt(mean_squared_error(y_test_lstm, y_pred_model_3))
r2_model_3 = r2_score(y_test_lstm, y_pred_model_3)

print('mean absolute error for LSTM:', mae_model_3)
print('root mean squared error for LSTM:', rmse_model_3)
print('r2 for LSTM:', r2_model_3)

##stacked lstm##

#stacked LSTM model building
# STACKED LSTM MODEL BUILDING

model_stacked_lstm = Sequential([

# DEFINES THE INPUT SHAPE FOR THE STACKED LSTM MODEL.
     Input(shape=(sequence_length, len(feature_columns))),

# FIRST LSTM LAYER LEARNS ENGINE PATTERNS FROM SEQUENCE DATA.
# RETURN_SEQUENCES=TRUE PASSES THE SEQUENCE TO THE NEXT LSTM LAYER.
     LSTM(64, return_sequences=True),

# DROPOUT LAYER TO REDUCE OVERFITTING BY RANDOMLY IGNORING
# 20% OF NEURONS DURING TRAINING.
     Dropout(0.2),

# SECOND LSTM LAYER LEARNS MORE PATTERNS FROM THE FIRST LSTM LAYER.
     LSTM(32),

# DROPOUT LAYER TO REDUCE OVERFITTING BY RANDOMLY IGNORING
# 20% OF NEURONS DURING TRAINING.
     Dropout(0.2),

# CREATES A LAYER WITH 32 NEURONS TO LEARN THE PATTERNS
# AND USE "RELU" TO CONVERT NEGATIVE VALUES TO ZERO.
     Dense(32, activation='relu'),

# CREATES A FINAL OUTPUT LAYER WITH ONE NEURON.
# IT COMBINES THE 32 VALUES FROM THE PREVIOUS LAYER USING LEARNED WEIGHTS AND BIAS
# AND OUTPUTS ONE PREDICTED RUL VALUE.
     Dense(1)
])


# TRAINING STACKED LSTM MODEL

model_stacked_lstm.compile(

# ADAM OPTIMIZER UPDATES THE MODEL WEIGHTS DURING TRAINING
# TO REDUCE THE PREDICTION ERROR.
     optimizer='adam',

# MODEL WILL USE "MSE" TO IMPROVE ITS PERFORMANCE.
     loss='mse',

# MAE SHOWS THE DIFFERENCE BETWEEN PREDICTED VALUES AND ACTUAL VALUES.
     metrics=['mae']
)

early_stopping_stacked = EarlyStopping(
     monitor='val_loss',
     patience=5,
     restore_best_weights=True
)

history_stacked_lstm = model_stacked_lstm.fit(

# TRAINING INPUT SEQUENCES.
     x_train_lstm,

# ACTUAL RUL VALUES FOR TRAINING THE MODEL.
     y_train_lstm,

# THE MODEL CAN GO THROUGH THE TRAINING DATA UP TO 50 TIMES.
# EARLY STOPPING MAY STOP TRAINING BEFORE 50 EPOCHS
# IF THE VALIDATION LOSS DOES NOT IMPROVE.
     epochs=50,

# THE MODEL UPDATES WEIGHTS AFTER PROCESSING 32 SAMPLES AT A TIME.
     batch_size=32,

# 20% OF THE TRAINING DATA IS USED FOR VALIDATION DURING TRAINING.
     validation_split=0.2,

# SHUFFLE IS FALSE BECAUSE LSTM USES SEQUENCE/TIME-BASED DATA.
# KEEPING THE ORDER HELPS PRESERVE THE SEQUENCE STRUCTURE.
     shuffle=False,

# STOPS TRAINING IF VALIDATION LOSS DOES NOT IMPROVE
# AND RESTORES THE BEST MODEL WEIGHTS.
     callbacks=[early_stopping_stacked]
)


# TESTING THE MODEL ON UNSEEN DATA

y_pred_stacked_lstm = model_stacked_lstm.predict(x_test_lstm).flatten()


# EVALUATION OF THE MODEL

mae_stacked_lstm = mean_absolute_error(y_test_lstm, y_pred_stacked_lstm)
rmse_stacked_lstm = np.sqrt(mean_squared_error(y_test_lstm, y_pred_stacked_lstm))
r2_stacked_lstm = r2_score(y_test_lstm, y_pred_stacked_lstm)

print('mean absolute error for Stacked LSTM:', mae_stacked_lstm)
print('root mean squared error for Stacked LSTM:', rmse_stacked_lstm)
print('r2 for Stacked LSTM:', r2_stacked_lstm)




# Bidirectional LSTM model building

model_bilstm = Sequential([
     Input(shape=(sequence_length, len(feature_columns))),

     Bidirectional(LSTM(64)),
     Dropout(0.2),

     Dense(32, activation='relu'),
     Dense(1)
])

# training Bidirectional LSTM model

model_bilstm.compile(
     optimizer='adam',
     loss='mse',
     metrics=['mae']
)

early_stopping_bilstm = EarlyStopping(
     monitor='val_loss',
     patience=5,
     restore_best_weights=True
)

history_bilstm = model_bilstm.fit(

# TRAINING INPUT SEQUENCES.
     x_train_lstm,

# ACTUAL RUL VALUES FOR TRAINING THE MODEL.
     y_train_lstm,

# THE MODEL CAN GO THROUGH THE TRAINING DATA UP TO 50 TIMES.
# EARLY STOPPING MAY STOP TRAINING BEFORE 50 EPOCHS
# IF THE VALIDATION LOSS DOES NOT IMPROVE.
     epochs=50,

# THE MODEL UPDATES WEIGHTS AFTER PROCESSING 32 SAMPLES AT A TIME.
     batch_size=32,

# 20% OF THE TRAINING DATA IS USED FOR VALIDATION DURING TRAINING.
     validation_split=0.2,

# SHUFFLE IS FALSE BECAUSE LSTM USES SEQUENCE/TIME-BASED DATA.
# KEEPING THE ORDER HELPS PRESERVE THE SEQUENCE STRUCTURE.
     shuffle=False,

# STOPS TRAINING IF VALIDATION LOSS DOES NOT IMPROVE
# AND RESTORES THE BEST MODEL WEIGHTS.
     callbacks=[early_stopping_bilstm]
)

# prediction

y_pred_bilstm = model_bilstm.predict(x_test_lstm).flatten()

# evaluation

mae_bilstm = mean_absolute_error(y_test_lstm, y_pred_bilstm)
rmse_bilstm = np.sqrt(mean_squared_error(y_test_lstm, y_pred_bilstm))
r2_bilstm = r2_score(y_test_lstm, y_pred_bilstm)

print('mean absolute error for Bidirectional LSTM:', mae_bilstm)
print('root mean squared error for Bidirectional LSTM:', rmse_bilstm)
print('r2 for Bidirectional LSTM:', r2_bilstm)

# check for maintenance
def maintenance_range(predicted_rul):
     if predicted_rul>=100:
          return "great","no risk"," No maintenance required"
     elif predicted_rul>=60:
          return "good"," medium risk","continue monitoring"
     elif predicted_rul>=30:
          return "not good"," high risk"," need maintenance"
     else:
          return "extremely bad"," very high risk"," immediate maintenance required"

# Empty list for storing the results
decision=[]

# temporarily storing the values using for loop
for actual_rul,predicted_rul in zip(y_test_lstm, y_pred_bilstm):
     status,risk,maintenance=maintenance_range(predicted_rul)
# values permanently stored in empty list
     decision.append([
          actual_rul,
          predicted_rul,
          status,
          risk,
          maintenance
     ])

# coverting the list into tabular format using pandas dataframe
decision_df=pd.DataFrame(
     decision,
     columns=[
          'actual_rul',
          'predicted_rul',
          'status',
          'risk',
          'maintenance'

     ])

# displaying the final results
print('Decision Support System Final Results')

print(decision_df.head(10))

# saving the file in csv format
decision_df.to_csv('decision_support_system_results.csv',index=False,sep=',')

print('decision support sytem results saved sucessfully')



