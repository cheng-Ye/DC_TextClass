
import  pandas as pd 
import matplotlib.pyplot as plt 
train=pd.read_csv('..\\new_data\\train_set.csv',usecols=['class'])
train['class'].value_counts().plot(kind='bar',title='the number of every class')
plt.show()
print(train['class'].unique())

print(len(train['class'].unique()))
