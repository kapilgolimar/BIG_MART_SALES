import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#impoting the datasets

train=pd.read_csv('train.csv')
test=pd.read_csv('test.csv')

#dependent and independent variable

y=train.iloc[:,[11]].values
x=train.iloc[:,[1,2,3,4,5,6,7,8,9,10]]
x.Outlet_Size=x.Outlet_Size.fillna(value='Medium')
x1=test.iloc[:,[1,2,3,4,5,6,7,8,9,10]]
x1.Outlet_Size=x1.Outlet_Size.fillna(value='Medium')

#filling missing values in weight column
x['new_weight']=x['Item_Weight'].groupby([x['Item_Type']]).apply(lambda z: z.fillna(z.mean()))
x1['new_weight']=x1['Item_Weight'].groupby([x1['Item_Type']]).apply(lambda z: z.fillna(z.mean()))
z=x.iloc[:,[1,2,4,6,7,8,9,10]]
z1=x1.iloc[:,[1,2,4,6,7,8,9,10]]

#replacing values in item type_fat_content
z.Item_Fat_Content=z.Item_Fat_Content.replace(['Low Fat', 'low fat'], 'LF')
z.Item_Fat_Content=z.Item_Fat_Content.replace(['Regular'], 'reg')
z1.Item_Fat_Content=z1.Item_Fat_Content.replace(['Low Fat', 'low fat'], 'LF')
z1.Item_Fat_Content=z1.Item_Fat_Content.replace(['Regular'], 'reg')

train_length = len(train)

dataset=pd.concat(objs=[z,z1],axis=0)
dataset = pd.get_dummies(dataset)
X=dataset.iloc[0:train_length,:].values
Z=dataset.iloc[train_length:14204,:].values

from sklearn.preprocessing import LabelEncoder,OneHotEncoder
labelencoder=LabelEncoder()
X[:,2]=labelencoder.fit_transform(X[:,2])
Z[:,2]=labelencoder.fit_transform(Z[:,2])

onehotencoder=OneHotEncoder(categorical_features=[2])
X=onehotencoder.fit_transform(X).toarray()
Z=onehotencoder.fit_transform(Z).toarray()

from sklearn.preprocessing import StandardScaler
fsx=StandardScaler()
fsy=StandardScaler()
X = fsx.fit_transform(X)
Z = fsx.transform(Z)
y = fsy.fit_transform(y)


from sklearn.ensemble import RandomForestRegressor
regressor=RandomForestRegressor(n_estimators=10,criterion='mse')
regressor=regressor.fit(X,y)

y_pred=regressor.predict(Z)
y_pred = fsy.inverse_transform(y_pred)
new_dataset=test_dataset.drop(['Item_Weight', 'Item_Visibility', 'Item_Fat_Content', 'Item_Type', 'Item_MRP', 'Outlet_Establishment_Year', 'Outlet_Size', 'Outlet_Location_Type', 'Outlet_Type'], axis=1)
new_dataset['Item_Outlet_Sales']=y_final

new_dataset.to_csv('C:/Users/bhupe/Desktop/Big Mart/Prt.csv',index=False)


