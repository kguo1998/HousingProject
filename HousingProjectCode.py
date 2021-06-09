import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import math

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
test_id = test['Id']
corrmat = train.corr()
corrmat['SalePrice'].sort_values(ascending=False)

#Creating new features: HasPorch, TotalSF, TotalBath, HasRemodel, SFDivRooms, MAYBE SFRatioAG, SFRatioBG, ExterQualRating, ExterCondRating
train['TotalSF'] = train['1stFlrSF'] + train['2ndFlrSF'] + train['TotalBsmtSF']
train['TotalBath'] = train['FullBath'] + (0.75*train['HalfBath']) + (0.56*train['BsmtFullBath']) + (0.42*train['BsmtHalfBath'])
train.loc[train['EnclosedPorch'] == 0, 'HasPorch'] = 0
train.loc[train['EnclosedPorch'] > 0, 'HasPorch'] = 1
train['SFDivRooms'] = train['GrLivArea']/train['TotRmsAbvGrd']
train.loc[(train['YearRemodAdd']-train['YearBuilt']) != 0, 'YearsAfterRemod'] = train['YearRemodAdd']-train['YearBuilt']
train.loc[(train['YrSold']-train['YearBuilt']) != 0, 'AgeSold'] = train['YrSold']-train['YearBuilt']
train.loc[(train['GrLivArea']+train['LotArea']) > 0 ,'LandAndBuildSF'] = train['GrLivArea']+train['LotArea']

#Testing new correlation
corrmat = train.corr()
corrmat['SalePrice'].sort_values(ascending=False)

#Adding in new features to train
columns = ['OverallQual','TotalSF','GrLivArea','TotalBath','SFDivRooms','GarageCars','GarageArea','TotalBsmtSF','1stFlrSF','FullBath','TotRmsAbvGrd','YearBuilt','YearRemodAdd','GarageYrBlt','MasVnrArea','Fireplaces','BsmtFinSF1','LotFrontage','WoodDeckSF','2ndFlrSF','OpenPorchSF','LandAndBuildSF','HalfBath','LotArea','BsmtFullBath','BsmtUnfSF','BedroomAbvGr','ScreenPorch','3SsnPorch','BsmtFinSF2','BsmtHalfBath','LowQualFinSF','OverallCond','EnclosedPorch','KitchenAbvGr','HasPorch','YearsAfterRemod','AgeSold']
train.update(train[columns].fillna(0))
added_features = ['TotalSF', 'TotalBath','YearsAfterRemod','AgeSold','SFDivRooms','HasPorch']
features = train.drop('SalePrice', axis = 1)
feature_list = list(features.columns)

#data is training set full 1460 rows with added features
data = train.loc[:, train.columns.isin(columns)]

#Creating new features for test
test['TotalSF'] = test['1stFlrSF'] + test['2ndFlrSF'] + test['TotalBsmtSF']
test['TotalBath'] = test['FullBath'] + (0.75*test['HalfBath']) + (0.5*test['BsmtFullBath']) + (0.25*test['BsmtHalfBath'])
test.loc[train['EnclosedPorch'] == 0, 'HasPorch'] = 0
test.loc[train['EnclosedPorch'] > 0, 'HasPorch'] = 1
test['SFDivRooms'] = test['GrLivArea']/test['TotRmsAbvGrd']
test.loc[(test['YearRemodAdd']-test['YearBuilt']) != 0, 'YearsAfterRemod'] = test['YearRemodAdd']-test['YearBuilt']
test.loc[(test['YrSold']-test['YearBuilt']) != 0, 'AgeSold'] = test['YrSold']-test['YearBuilt']
test.loc[(test['GrLivArea']+test['LotArea']) > 0 ,'LandAndBuildSF'] = test['GrLivArea']+test['LotArea']

#test set
test.update(test[columns].fillna(0))
#data2 is test set rows with added features
data2 = test.loc[:, test.columns.isin(columns)]

#Evaluating original model accuracy
original_feature_indices = [feature_list.index(feature) for feature in feature_list if feature in columns]
original_test_features = train.iloc[:, original_feature_indices]
original_test_features = original_test_features.drop(added_features, axis=1)

#Splitting data into 'training' and 'test' set
training_set = data.loc[0:1168,:]
test_set = data.loc[1168:1460,:]


sp = np.array(train['SalePrice'])

y_train = train['SalePrice']

rf_regr = RandomForestRegressor(max_depth=8, random_state=0)

rf_regr.fit(training_set, y_train[0:1169])

y_pred2 = rf_regr.predict(test_set)

#taking training set model and predicting whole data
y_pred_final = rf_regr.predict(data2)

#Original data model fitting and predicting
rf_regr.fit(original_test_features.loc[0:1168,:], y_train[0:1169])

y_pred_orig = rf_regr.predict(original_test_features.loc[1168:1460,:])

#Establishing baseline
#Establishing error
errors = abs(y_pred_orig - y_train[1168:1460])
print('Average absolute error:', round(np.mean(errors), 2), ' degrees.')
mape = 100* (errors/y_train[1168:1460])
accuracy = 100 - np.mean(mape)
print('Accuracy:', round(accuracy, 2), '%.')
rmse = math.sqrt(np.mean((y_pred_orig - y_train[1168:1460])**2))
rms = mean_squared_error(y_train[1168:1460], y_pred_orig, squared=False)
print('RMSE:', rmse)
#print('sklearnRSME:',rms)

#Establishing error of model with added parameters
errors2 = abs(y_pred2 - y_train[1168:1460])
print('Average absolute error2:', round(np.mean(errors2), 2), ' degrees.')
mape2 = 100* (errors2/y_train[1168:1460])
accuracy2 = 100 - np.mean(mape2)
print('Accuracy2:', round(accuracy2, 2), '%.')
rmse2 = math.sqrt(np.mean((y_pred2 - y_train[1168:1460])**2))
print("RMSE2:", rmse2)

#Exporting file with Id to csv file
sub = pd.DataFrame()
sub['Id'] = test_id
sub['SalePrice'] = y_pred_final
sub.to_csv('submission7RF.csv', index=False)