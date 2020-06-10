from sklearn.datasets import load_digits
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import pandas as pd
import numpy as np

data=load_digits()
x=data.images.reshape(len(data.images),-1)
y=data.target

X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.3)
print(len(y_test))
model=model=MLPClassifier(hidden_layer_sizes=(500,500,))
model.fit(X_train,y_train)
y_pred=model.predict(X_test)
print(accuracy_score(y_pred,y_test))
#print(pd.DataFrame(model.predict_proba(X_test)).head)

smvalue=np.array(model.predict_proba(X_test))
tmp=[]
outvalue=[]
for i in range(len(y_test)):
    if y_test[i]!=y_pred[i]:
        tmp.append(i)
        tmp.append(y_test[i])
        tmp.append(y_pred[i])
        tmp.append(max(smvalue[i]))
        outvalue.append(tmp)
        tmp=[]
print(outvalue)

labels = sorted(list(set(y_test)))
cmx_data = confusion_matrix(y_test, y_pred, labels=labels)
df_cmx = pd.DataFrame(cmx_data, index=labels, columns=labels)
plt.figure(figsize = (10,7))
sns.heatmap(df_cmx, annot=True)
plt.show()
