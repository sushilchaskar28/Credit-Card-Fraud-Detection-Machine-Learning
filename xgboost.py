#Importing libraries
import pandas as pd
import numpy as np
import random
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier as GBC
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings("ignore")

def getfscore(cm):
    TP=cm[0][0]
    TN=cm[1][1]
    FN=cm[0][1]
    FP=cm[1][0]
    precision = TP/(TP+FP)
    recall = TP/(TP+FN)
    fmesr=(2*precision*recall)/(precision+recall)
    return fmesr 
 
#loading Dataset 
dataset = pd.read_csv("creditcard.csv")
colName = [i for i in dataset.columns] #columns
colName = colName[1:-1]
dataset = dataset.sample(frac = 1) #shuffle dataset
x = dataset.iloc[:,1:-1].values.tolist() #features
y = dataset.iloc[:,-1].values.tolist() #labels

negData = y.count(0)
posData = y.count(1)
negPer = (negData/len(y))*100
posPer = (posData/len(y))*100
print("Fraud data percentage: {0}".format(posPer))
print("Dataset is highly unbalanced. We will have to balance the dataset")

x_neg = dataset.loc[dataset['Class']==0].sample(frac=1)
x_pos = dataset.loc[dataset['Class']==1].sample(frac=1)

#Geneate 5 random datasets
xnegArr=[]
for i in range(5):
    k = random.randrange(negData) - posData
    xnegArr.append(x_neg[k:k+posData])
new_dsArr=[]
for i in range(5):
    new_dsArr.append(pd.concat([x_pos, xnegArr[i]]).sample(frac = 1))
    
print("Method 5: training using XGBoost....")   
 
x_set = new_dsArr[0].iloc[:,1:-1].values.tolist()
y_set = new_dsArr[0].iloc[:,-1].values.tolist()
x_train, x_test, y_train, y_test = train_test_split(x_set, y_set, test_size=0.30, random_state=1) #training on balanced dataset 1
cls_rf = GBC(learning_rate=0.12,n_estimators=120)
cls_rf.fit(x_train, y_train)
y_pred = cls_rf.predict(x_test)
cm = confusion_matrix(y_test, y_pred)
print("For dataset 1, f-measure score is: {0}".format(getfscore(cm)))
print("")
print("using classifier on remaining test datasets")
for i in range(1,5):
    #x_set = preprocessing.scale(new_dsArr[i].iloc[:,1:-1].values.tolist()) #for logistic Reg
    x_set = new_dsArr[i].iloc[:,1:-1].values.tolist()
    y_set = new_dsArr[i].iloc[:,-1].values.tolist()
    y_pred = cls_rf.predict(x_set)
    cm = confusion_matrix(y_set, y_pred)
    print("For dataset {0}, f-measure score is: {1}".format(i+1,getfscore(cm)))

ds = pd.concat([x_pos[0:350], x_neg[0:190000]]).sample(frac = 1) #training set
ds_test = pd.concat([x_pos[350:], x_neg[190000:]]).sample(frac = 1) # test set
x_train = (ds.iloc[:,1:-1].values.tolist())
y_train = ds.iloc[:,-1].values.tolist()
x_test = ds_test.iloc[:,1:-1].values.tolist()
y_test= ds_test.iloc[:,-1].values.tolist()
y_pred = cls_rf.predict(x_train)
cm = confusion_matrix(y_train, y_pred)
print("For train dataset, f-measure score is: {0}".format(getfscore(cm)))

y_pred = cls_rf.predict(x_test)
cm = confusion_matrix(y_test, y_pred)
print("For test dataset, f-measure score is: {0}".format(getfscore(cm)))
print("")
print("testing on complete dataset")
y_pred = cls_rf.predict(x)
cm = confusion_matrix(y, y_pred)
print("For original dataset, f-measure score is: {0}".format(getfscore(cm)))
print("")

#Feature selection
print("feature selection")
fea_sc= (cls_rf.feature_importances_)
fea_arr = sorted(zip(map(lambda x: round(x, 5), fea_sc), colName), reverse=True)
felen= len(colName)
colDel = [fea_arr[i][1] for i in range(17,felen)]
modDataset=[]
for i in range(5):
    modDataset.append(new_dsArr[i].drop(colDel, axis=1))
x_set = modDataset[0].iloc[:,1:-1].values.tolist()
y_set = modDataset[0].iloc[:,-1].values.tolist()
x_train, x_test, y_train, y_test = train_test_split(x_set, y_set, test_size=0.30, random_state=1)
clsf = GBC(learning_rate=0.12,n_estimators=120)
clsf.fit(x_train, y_train)
y_pred = clsf.predict(x_test)
cm = confusion_matrix(y_test, y_pred)
print("For dataset {0}, f-measure score after feature selection is: {1}".format(1, getfscore(cm)))

print("testing on original dataset")
dataNew = dataset.drop(colDel, axis=1)
x = dataNew.iloc[:,1:-1].values.tolist() #features
y = dataNew.iloc[:,-1].values.tolist() #labels
x_neg = dataNew.loc[dataset['Class']==0].sample(frac=1)
x_pos = dataNew.loc[dataset['Class']==1].sample(frac=1)
ds = pd.concat([x_pos[0:350], x_neg[0:190000]]).sample(frac = 1) #training set after feature selection
ds_test = pd.concat([x_pos[350:], x_neg[190000:]]).sample(frac = 1) #test set after feature selection
x_train = (ds.iloc[:,1:-1].values.tolist())
y_train = ds.iloc[:,-1].values.tolist()
x_test = ds_test.iloc[:,1:-1].values.tolist()
y_test= ds_test.iloc[:,-1].values.tolist()
y_pred = clsf.predict(x_train)
cm = confusion_matrix(y_train, y_pred)
print("For train dataset, f-measure score is: {0}".format(getfscore(cm)))

y_pred = clsf.predict(x_test)
cm = confusion_matrix(y_test, y_pred)
print("For test dataset, f-measure score is: {0}".format(getfscore(cm)))
print("")
print("testing on complete dataset")
y_pred = clsf.predict(x)
cm = confusion_matrix(y, y_pred)
print("For original dataset, f-measure score is: {0}".format(getfscore(cm)))