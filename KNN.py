# # Load Dataset
import pandas as pd

df = pd.read_csv('data.csv',header=None)
print(df)


# # Preprocessing
X = []
y = []
for (value,label) in zip(df[1],df[2]):
    X.append([value])
    y.append([label])


# # Create Model
from sklearn.neighbors import KNeighborsClassifier

n_neighbors = 3

clf = KNeighborsClassifier(n_neighbors)

clf.fit(X,y)


# # Testing Model
datatest = pd.read_csv('testing.csv',header=None)

for (laptop,harga) in zip(datatest[0],datatest[1]):
    predict = clf.predict([[harga]])
    print("laptop {laptop} dengan harga {harga} itu {hasil}".format(laptop=laptop,harga=harga,hasil=predict[0]))

