import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
import numpy as np

df = pd.read_csv('train.csv')
df2 = pd.read_csv('test.csv')
print(df.info())
for i in df:
    df[i] = df[i].fillna(0)
print(df.head())

# d2 = (df.groupby(by = 'occupation_type')['result'].value_counts())
# d2.plot(kind = 'pie')
# plt.show()


# d1 = df.groupby(by = 'life_main')['result'].value_counts() 
# print(d1)
# d = df.pivot_table(index = 'life_main', columns = 'occupation_type', values = 'result', aggfunc = 'mean')
# d.plot(kind = 'barh',subplots = True ) 
# plt.show()  

def reverce_value(value):
    if value == 'False':
        return 0
    return 1
df['career_start'] = df['career_start'].apply(reverce_value)
df['career_end'] = df['career_end'].apply(reverce_value)
d = df.groupby(by = 'career_end')['result'].value_counts()
d.plot(kind = 'pie')
plt.show()  

def occupation_type(value):
    if value == 'university':
        return 1
    return 2
df['occupation_type'] = df['occupation_type'].apply(occupation_type)
def life_main(value):
    if value != 'False':
        return int(value)
    return 0
# df['life_main'] = df['life_main'].apply(life_main)
print(df.info())
df.drop([
        'id','sex','bdate',
        'has_photo','has_mobile','followers_count',
        'graduation','relation','education_form',
        'education_status','langs','people_main',
        'city','last_seen','occupation_name','life_main'
        ], axis = 1, inplace = True)
x = df.drop('result',axis = 1)
y = df['result']      
x_test, x_train, y_test, y_train = train_test_split(x,y,test_size = 0.5)
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)
classifier = KNeighborsClassifier(n_neighbors = 35)
classifier.fit(x_train,y_train)  
y_pred = classifier.predict(x_test)
print(accuracy_score(y_test,y_pred)*100) 
print(confusion_matrix(y_test,y_pred))
# error_rates = []
# for i in np.arange(1, 101): 
#     classifier = KNeighborsClassifier(n_neighbors = i)
#     classifier.fit(x_train, y_train)
#     new_predictions = classifier.predict(x_test)
#     error_rates.append(np.mean(new_predictions != y_test))
# plt.plot(error_rates)
# plt.show() 
