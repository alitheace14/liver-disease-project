import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report,confusion_matrix,ConfusionMatrixDisplay
df = pd.read_csv("c:/Users/nsr/Downloads/Telegram Desktop/archive/liverdataset.csv", encoding='windows-1252')
df=df.dropna()
df.columns = df.columns.str.strip()
print(df.columns)
scaler=StandardScaler()
df['Gender of the patient']=df['Gender of the patient'].apply(lambda x: 1 if x == 'male' else 0)
x=df[['Age of the patient', 'Gender of the patient', 'Total Bilirubin',
       'Direct Bilirubin', 'Alkphos Alkaline Phosphotase',
       'Sgpt Alamine Aminotransferase', 'Sgot Aspartate Aminotransferase',
       'Total Protiens', 'ALB Albumin', 'A/G Ratio Albumin and Globulin Ratio']]
y=df['Result']
x=scaler.fit_transform(x)
train_x,test_x,train_y,test_y=train_test_split(x,y,test_size=0.2,random_state=12)
forest=RandomForestClassifier(n_estimators=25,random_state=12).fit(train_x,train_y)
prediction=forest.predict(test_x)
print(classification_report(train_y,forest.predict(train_x)))
print(classification_report(test_y,prediction))

conf=confusion_matrix(test_y,prediction,labels=[1,2])
np.set_printoptions(precision=2)
display=ConfusionMatrixDisplay(confusion_matrix=conf,display_labels=[1,2])
display.plot()
plt.show()