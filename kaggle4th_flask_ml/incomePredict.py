import pandas as pd
import numpy as np
# import seaborn as sns
import sklearn as sk
import flask

print(pd.__version__)
print(np.__version__)
print(sk.__version__)



print(flask.__version__)

from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

import os

print(os.getcwd())
path ='C:\\Users\\KOREA\Documents\\GitHub\\ML_LIB_CLASS\\kaggle4th_flask_ml\\'
train =pd.read_csv(path +'data\\train.csv')
test= pd.read_csv(path +'data\\test.csv')
sub = pd.read_csv(path +'data\\sample_submission.csv')

print(train.shape, test.shape, sub.shape)

train.loc[train['income']=='>50K', 'target'] =1
train.loc[train['income']=='<=50K', 'target'] =0

train['target'] =train['target'].astype('int')
sel =['age', 'fnlwgt','education_num','hours_per_week','capital_gain','capital_loss']
X = train[sel]
y=train['target']
test_X = test[sel]

X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                stratify= train.target, random_state=42)
model =RandomForestClassifier(random_state=30).fit(X_train, y_train)
score =cross_val_score(model,X_test,y_test, cv=5,scoring='roc_auc')
print("교차 검증 점수(AUC) - rf : ", np.mean(score))

from sklearn.linear_model import LogisticRegression
model = GradientBoostingClassifier(random_state=30).fit(X_train, y_train)
score =cross_val_score(model,X_test,y_test, cv=5,scoring='roc_auc')
print("교차 검증 점수(AUC) - rf : ", np.mean(score))
import pickle
pickle.dump(model, open(path + "\\model\\income_base.pkl", 'wb'))
train.describe()
