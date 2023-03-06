import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle as pk

dataset = pd.read_csv('iris.csv')

# print(dataset.head())

X = dataset.drop(columns='Class', axis=1)
Y = dataset['Class']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=2)

sc = StandardScaler()

X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

classifier = RandomForestClassifier()
classifier.fit(X_train, Y_train)

pk.dump(classifier, open('model.pkl', 'wb'))

print('Done!')